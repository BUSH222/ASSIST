"""`assist profile build`: runs every survey step and writes profiles/vhf137/.

Outputs in `out_dir` (committed, except bank/): profile.json, noise_psd.npy, noise_hourly.npy, birdies.json,
orbcomm_ref.json, meteor_cal.json, bank/. Intermediates and the event list for labelling go to `work_dir`
(tool outputs, e.g. out/profile/). The profile holds no timestamps and only the rounded station position, so a
rebuild from the same inputs gives the same files.
"""

import csv
import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from assist import waterfall as wf
from assist.orbits.elements import load_newest
from assist.orbits.geometry import find_passes
from assist.profile import adc, birdies, humps, labels, meteor_cal, mirror, noise, occupancy, orbcomm
from assist.profile.annotations import excluded_ranges, keep_mask, load_annotations
from assist.profile.bank import Bank, strongest_bursts
from assist.profile.cache import day_summary, wav_rows
from assist.profile.rawio import RawWaterfall, open_raw
from assist.profile.wavio import Recording
from assist.station import load_station

if TYPE_CHECKING:
    from collections.abc import Callable

FC = 137.0e6
METEORS = {57166: "METEOR-M2 3", 59051: "METEOR-M2 4"}
LORA_CHANNELS_HZ = ((137.036e6, 137.162e6), (137.863e6, 137.990e6))  # SpaceBEE; label guess only


def _utc(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _jsonable(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, 6)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=False) + "\n")


def _row_times(raw: RawWaterfall) -> np.ndarray:
    return np.asarray(raw.row_time(np.arange(raw.n_rows)), dtype=np.float64)


def _hourly_counts(t_unix: np.ndarray, n_days: int) -> list[float]:
    h = ((np.asarray(t_unix) % 86400) // 3600).astype(int)
    return [float(v) for v in np.bincount(h, minlength=24) / max(1, n_days)]


def _merge_birdies(per_day: list[list[dict[str, Any]]], tol_hz: float = 300.0) -> list[dict[str, Any]]:
    """Birdies found on every day (frequencies within tol), averaged."""
    out = []
    for b in per_day[0]:
        matches = [min(d, key=lambda c: abs(c["freq_hz"] - b["freq_hz"])) for d in per_day[1:] if d]
        if all(abs(m["freq_hz"] - b["freq_hz"]) <= tol_hz for m in matches) and len(matches) == len(per_day) - 1:
            grp = [b, *matches]
            out.append(
                {
                    "freq_hz": float(np.mean([g["freq_hz"] for g in grp])),
                    "excess_db": float(np.mean([g["excess_db_median"] for g in grp])),
                    "width_bins": int(max(g["width_bins"] for g in grp)),
                    "fraction_min": float(min(g["fraction"] for g in grp)),
                }
            )
    return out


def build(
    wav: Path,
    raw_paths: list[Path],
    station_path: Path,
    element_paths: list[Path],
    annotation_paths: list[Path],
    saw_path: Path,
    out_dir: Path,
    work_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / "cache"
    st = load_station(station_path)
    ann = load_annotations(list(annotation_paths))
    els = load_newest(list(element_paths))
    rec = Recording(wav, tz=st.tz)
    raws = [open_raw(p) for p in sorted(raw_paths)]
    f = wf.bin_freqs(FC)
    fm = f / 1e6
    saw = {float(r["freq_hz"]): float(r["s21_db"]) for r in csv.DictReader(saw_path.open())}

    prof: dict[str, Any] = {
        "version": 1,
        "band": "vhf137",
        "fc_hz": FC,
        "fs_hz": wf.FS,
        "fft_size": wf.FFT_SIZE,
        "station": {"lat_lon_rounded": [round(st.lat_deg, 1), round(st.lon_deg, 1)], "alt_m": st.alt_m, "tz": st.tz},
        "inputs": {
            "recording": {"file": wav.name, "bytes": wav.stat().st_size},
            "waterfalls": [{"file": r.path.name, "bytes": r.path.stat().st_size} for r in raws],
            "elements": [{"file": Path(p).name, "bytes": Path(p).stat().st_size} for p in element_paths],
            "saw": {"file": saw_path.name, "points": len(saw)},
        },
        "annotations": [a.as_dict() for a in ann],
    }

    log("recording and waterfall timing")
    i = rec.info
    prof["recording"] = {
        "start_utc": _utc(rec.start_unix),
        "duration_s": rec.duration_s,
        "n_samples": rec.n_samples,
        "riff": "RF64" if i.rf64 else "RIFF",
        "size_fields_wrong": i.size_fields_wrong,
        "auxi": i.auxi is not None,
        "start_time_source": "SDR++ file name (local time " + st.tz + ")",
    }
    prof["waterfalls"] = [
        {
            "file": r.path.name,
            "rows": r.n_rows,
            "first_row_utc": _utc(float(r.row_time(0))),
            "last_row_utc": _utc(float(r.row_time(r.n_rows - 1))),
        }
        for r in raws
    ]

    log("ADC statistics")
    f_adc = cache / f"adc_{wav.stem}_{wav.stat().st_size}.npz"
    if f_adc.exists():
        stats = dict(np.load(f_adc))
    else:
        stats = adc.per_second_stats(rec)
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(f_adc, t0=stats["t0"], hist_i=stats["hist_i"], hist_q=stats["hist_q"], cov_iq=stats["cov_iq"])
    ex_noise = excluded_ranges(ann, wav, "noise") + excluded_ranges(ann, wav, "adc")
    prof["adc"] = adc.summarise(stats, ex_noise)
    sigma = float(prof["adc"]["noise_sigma_mad_counts"])

    log("Waterfall rows of the recording and day summaries")
    rows, _min_db = wav_rows(rec, cache)
    t_r0 = (wf.CAL_SAMPLES + np.arange(len(rows)) * wf.ROW_SAMPLES) / rec.fs
    t_r1 = t_r0 + wf.ROW_SECONDS
    keep_n = keep_mask(t_r0, ex_noise) & keep_mask(t_r1, ex_noise)
    wav_shape = np.median(rows[keep_n], axis=0)
    days = [day_summary(r, cache) for r in raws]

    log("birdies")
    per_day = [birdies.find_birdies(d["minute_mean_db"], f) for d in days]
    vdl_bins = (fm > 136.9) & (fm < 136.95)
    bl = [b for b in _merge_birdies(per_day) if not vdl_bins[np.argmin(np.abs(f - b["freq_hz"]))]]
    bw = birdies.find_birdies(rows[keep_n], f, threshold_db=1.0)
    _write_json(
        out_dir / "birdies.json", {"method": "per-minute means, > 0.5 dB over local median in > 90 %", "birdies": bl}
    )
    prof["birdies"] = {"count": len(bl), "file": "birdies.json", "recording_rows_1db": [b["freq_hz"] for b in bw]}

    log("noise PSD, DC, diurnal curve, regimes")
    shapes = [noise.day_shape(d["minute_mean_db"]) for d in days]
    raw_shape, clipped = noise.combined_shape(shapes, wav_shape, fm)
    bmask = np.zeros(len(f), bool)
    for b in bl:
        k = int(np.argmin(np.abs(f - b["freq_hz"])))
        bmask[max(0, k - b["width_bins"] // 2 - 2) : k + b["width_bins"] // 2 + 3] = True
    psd = noise.smooth_shape(raw_shape, fm, bmask).astype(np.float32)
    np.save(out_dir / "noise_psd.npy", psd)
    dcs = [noise.dc_spike(s) for s in shapes]
    dc_db = float(np.mean([d["centre_bin_excess_db"] for d in dcs]))
    flat = noise.flat_mask(fm)
    wav_n = noise.normalise(wav_shape, fm)
    diff = (wav_n - noise.normalise(np.mean(shapes, axis=0), fm))[flat]
    hourly = np.nanmean(
        [noise.hourly_curve(d["row_level_db"], _row_times(r)) for d, r in zip(days, raws, strict=True)], axis=0
    )
    np.save(out_dir / "noise_hourly.npy", hourly.astype(np.float32))
    regimes = []
    for d in days:
        lvl = d["row_level_db"]
        regimes.append(noise.regimes(lvl[: len(lvl) // 60 * 60].reshape(-1, 60).mean(axis=1)))
    prof["noise"] = {
        "psd_file": "noise_psd.npy",
        "psd_reference": "dB, 0 = median 136.2-137.8 MHz (VDL excluded); ~4.5 kHz smoothing; DC, birdies removed",
        "psd_source": "median of per-minute means of both .raw days; recording shape spliced where days clip",
        "clipped_bins_from_recording": int(clipped.sum()),
        "recording_vs_days_shape_rms_db": float(np.sqrt(np.mean((diff - diff.mean()) ** 2))),
        "days_shape_rms_db": float(
            np.std(noise.normalise(shapes[0], fm)[flat] - noise.normalise(shapes[-1], fm)[flat])
        ),
        "dc_spike": {
            "centre_bin_excess_db": dc_db,
            "per_day": dcs,
            "amplitude_counts_est": noise.dc_amplitude_counts(dc_db, sigma),
            "recording": noise.dc_spike(wav_shape),
            "note": "recording shows a DC notch instead (DC removed before recording)",
        },
        "sigma_counts": sigma,
        "hourly_file": "noise_hourly.npy",
        "hourly_utc_db": hourly.tolist(),
        "regimes_per_day": regimes,
    }

    log("image / mirror tests")
    srcs = [(136.925e6, 12e3), (136.975e6, 10e3), (137.9e6, 55e3)]
    img = {}
    for name, spec in [
        *((r.path.name, d["minute_mean_db"]) for r, d in zip(raws, days, strict=True)),
        (wav.name, rows),
    ]:
        img[name] = [mirror.image_rejection(spec.astype(np.float64), f, s, bw_) for s, bw_ in srcs]
    n0 = int(600 * rec.fs)
    coh = [mirror.coherent_image(rec.iq, n0, n0 + int(60 * rec.fs), rec.fs, off) for off in (-75e3, -25e3)]
    # 138 MHz carrier (strongest row of the recording) against its mirror at 2 fc - f
    band = (f > 138.0e6) & (f < 138.2e6)
    ex138 = rows[:, band].max(axis=1) - np.median(rows[:, band], axis=1)
    i138 = int(np.argmax(ex138))
    f138 = float(f[np.flatnonzero(band)[int(np.argmax(rows[i138, band]))]])
    n138 = int(wf.CAL_SAMPLES + i138 * wf.ROW_SAMPLES)
    coh138 = mirror.coherent_image(rec.iq, n138, n138 + wf.ROW_SAMPLES, rec.fs, f138 - FC)
    coh138["row_excess_db"] = float(ex138[i138])
    prof["mirror"] = {
        "image_vs_source_minute_scale": img,
        "coherent_iq_image_recording": coh,
        "carrier138_vs_mirror_recording": coh138,
    }

    log("occupancy")
    chans = occupancy.default_channels()
    bird_f = np.array([b["freq_hz"] for b in bl])
    occ_days = []
    event_rows: list[dict[str, Any]] = []
    for r, d in zip(raws, days, strict=True):
        f_occ = cache / f"occ_{r.path.stem.split('_')[1]}.npz"
        if f_occ.exists():
            z = np.load(f_occ)
            exc, peak, base = z["exc"], z["peak"], z["base"]
        else:
            base = occupancy.hourly_baseline(d["minute_mean_db"])
            exc, peak = occupancy.day_channel_excess(r, base, chans, f)
            np.savez(f_occ, exc=exc, peak=peak, base=base)
        occ_days.append((r, exc, peak, base))
    rate_per_chan: dict[str, dict[str, Any]] = {}
    for j, c in enumerate(chans):
        evs = []
        for r, exc, peak, _b in occ_days:
            ev = occupancy.events_from_series(exc[:, j], 6.0, 3.0)
            t = np.asarray(r.row_time(ev[:, 0])) if len(ev) else np.zeros(0)
            pk = np.array([float(exc[s:e, j].max()) for s, e in ev]) if len(ev) else np.zeros(0)
            fpk = (
                np.array([float(f[peak[s + int(np.argmax(exc[s:e, j])), j]]) for s, e in ev])
                if len(ev)
                else np.zeros(0)
            )
            evs.append((r, ev, t, pk, fpk))
        n = [len(e[1]) for e in evs]
        if sum(n) == 0:
            continue
        dur = np.concatenate([(e[1][:, 1] - e[1][:, 0]) * wf.ROW_SECONDS for e in evs if len(e[1])])
        rate_per_chan[c.name] = {
            "kind": c.kind,
            "centre_hz": c.centre_hz,
            "events_per_day": n,
            "duty": [float((e[1][:, 1] - e[1][:, 0]).sum() / len(occ_days[0][1])) if len(e[1]) else 0.0 for e in evs],
            "_t": np.concatenate([e[2] for e in evs]),
            "_dur": dur,
            "_pk": np.concatenate([e[3] for e in evs]),
            "_f": np.concatenate([e[4] for e in evs]),
        }
        on_birdie = bool(len(bird_f)) and float(np.min(np.abs(bird_f - c.centre_hz))) < 4e3
        for r, ev, t, pk, fpk in evs:
            for (s, e), tt, p_, ff in zip(ev, t, pk, fpk, strict=True):
                auto = c.kind if c.kind != "band138" else "carrier_138"
                if c.kind == "am" and (e - s) * wf.ROW_SECONDS > 60:
                    auto = "unknown"
                if c.kind == "am" and on_birdie:
                    auto = "birdie"
                event_rows.append(
                    {
                        "file": r.path.name,
                        "row_start": int(s),
                        "row_end": int(e),
                        "utc_start": _utc(float(tt)),
                        "utc_end": _utc(float(r.row_time(e))),
                        "f_lo_hz": float(ff - 5e3) if c.kind == "band138" else c.f_lo_hz,
                        "f_hi_hz": float(ff + 5e3) if c.kind == "band138" else c.f_hi_hz,
                        "auto_label": auto,
                        "peak_db": round(float(p_), 2),
                    }
                )
    n_days = len(raws)

    def stats(names: list[str], max_dur: float = np.inf) -> dict[str, Any]:
        t = np.concatenate([rate_per_chan[n]["_t"] for n in names]) if names else np.zeros(0)
        dur = np.concatenate([rate_per_chan[n]["_dur"] for n in names]) if names else np.zeros(0)
        pk = np.concatenate([rate_per_chan[n]["_pk"] for n in names]) if names else np.zeros(0)
        sel = dur <= max_dur
        return {
            "events_per_day": float(sel.sum() / n_days),
            "events_per_hour_utc": _hourly_counts(t[sel], n_days),
            "duration_s": occupancy.lognormal_fit(dur[sel]),
            "peak_excess_db_p10_p50_p90": [float(v) for v in np.percentile(pk[sel], [10, 50, 90])]
            if sel.any()
            else [],
        }

    vdl_active = [n for n, v in rate_per_chan.items() if v["kind"] == "vdl" and min(v["events_per_day"]) >= 1000]
    am_ok = [
        n
        for n, v in rate_per_chan.items()
        if v["kind"] == "am" and not (len(bird_f) and np.min(np.abs(bird_f - v["centre_hz"])) < 4e3)
    ]
    am_excluded = sorted(n for n, v in rate_per_chan.items() if v["kind"] == "am" and n not in am_ok)
    starts, power = _wav_stft(rec, cache, chans, excluded_ranges(ann, wav, "occupancy"))
    frame_s = 1024 / rec.fs
    vdl_names = [c.name for c in chans if c.kind == "vdl"]
    wav_b = {n: occupancy.bursts(power[:, vdl_names.index(n)], frame_s) for n in vdl_active}
    b138 = rate_per_chan.get("band_138.0-138.2")
    prof["occupancy"] = {
        "rows_on_off_db": [6.0, 3.0],
        "row_s": wf.ROW_SECONDS,
        "vdl": {
            n: {
                "events_per_day": rate_per_chan[n]["events_per_day"],
                "duty_rows": rate_per_chan[n]["duty"],
                **stats([n]),
                "recording_fine_time": wav_b[n],
            }
            for n in vdl_active
        },
        "vdl_spill_channels": sorted(
            n
            for n, v in rate_per_chan.items()
            if v["kind"] == "vdl" and n not in vdl_active and max(v["events_per_day"]) >= 100
        ),
        "am": {
            "voice_like_max_duration_s": 60,
            **stats(am_ok, 60.0),
            "top_channels": sorted(
                ((n, float(np.mean(rate_per_chan[n]["events_per_day"]))) for n in am_ok), key=lambda x: -x[1]
            )[:10],
            "excluded_birdie_channels": am_excluded,
            "long_carriers_per_day": stats(am_ok)["events_per_day"] - stats(am_ok, 60.0)["events_per_day"],
        },
        "band138": None
        if b138 is None
        else {
            **stats(["band_138.0-138.2"]),
            "freq_mhz_p10_p50_p90": [float(v) / 1e6 for v in np.percentile(b138["_f"], [10, 50, 90])],
            "modulation": "unmodulated carriers (recording: < 25 Hz wide, steady envelope)",
        },
    }

    # mirror tests continued: 138 MHz events vs their mirror (135.8-136.0 MHz)
    if b138 is not None:
        prof["mirror"]["band138_events_vs_mirror"] = _band138_mirror(occ_days, chans, f)

    log("humps and wandering lines")
    step_s = humps.STEP_ROWS * wf.ROW_SECONDS
    fcol = f.reshape(-1, humps.COL_BINS).mean(axis=1) / 1e6
    h_out = {}
    for r, _exc, _pk, base in occ_days:
        fc_ = cache / f"humps_{r.path.stem.split('_')[1]}.npy"
        if fc_.exists():
            cube = np.load(fc_)
        else:
            cube = humps.day_cube(r, base)
            np.save(fc_, cube)
        lines = humps.wandering_lines(humps.track(humps.detect(cube), step_s), step_s, fcol)
        h_out[r.path.name] = {
            "comb": humps.comb_summary(humps.comb_series(cube, fcol), step_s),
            "wandering_lines": lines,
        }
        for c_ in _comb_episodes(humps.comb_series(cube, fcol), step_s):
            s, e = c_
            event_rows.append(
                _row_event(r, s * humps.STEP_ROWS, e * humps.STEP_ROWS, 136.1e6, 137.84e6, "usb_hump", 0.0)
            )
        for ln in lines:
            s = ln["start_step"] * humps.STEP_ROWS
            e = s + int(ln["duration_s"] / wf.ROW_SECONDS)
            event_rows.append(
                _row_event(r, s, e, ln["f_min_mhz"] * 1e6 - 5e3, ln["f_max_mhz"] * 1e6 + 5e3, "unknown", 0.0)
            )
    wcube = humps.residual_cube(rows.copy(), np.median(rows[keep_n], axis=0), fm)[int(445 / step_s) + 1 :]
    h_out[wav.name] = {"comb": humps.comb_summary(humps.comb_series(wcube, fcol), step_s), "note": "after 445 s"}
    prof["humps"] = h_out

    log("METEOR level calibration")
    pds = meteor_cal.collect(raws, els, st, tuple(METEORS))
    use = [p for p in pds if p.p.max_el_deg > 5]
    fits = {METEORS[n]: meteor_cal.fit_c([p for p in use if p.norad == n], st) for n in METEORS}
    both = meteor_cal.fit_c(use, st)
    loss = -saw[min(saw, key=lambda k: abs(k - meteor_cal.F_METEOR))]
    frac = meteor_cal.band_power_fraction()
    edges = np.array([meteor_cal.edge_offsets(p) for p in use])
    mcal = {
        "method": meteor_cal.__doc__,
        "passes_measured": len(pds),
        "passes_used_max_el_gt_5": len(use),
        "fit_per_satellite": fits,
        "fit_both": both,
        "residual_tables": meteor_cal.residual_table(use, both["C_db"], st),
        "edges_s": {
            "rise_minus_aos_p10_p50_p90": [float(v) for v in np.nanpercentile(edges[:, 0], [10, 50, 90])],
            "los_minus_fall_p10_p50_p90": [float(v) for v in np.nanpercentile(edges[:, 1], [10, 50, 90])],
        },
        "band_power_fraction": frac,
        "saw_loss_db_137_9": loss,
        "p_tx_dbw_for_t_ant": {
            str(t): {
                "t_sys_k": meteor_cal.t_sys_k(t, loss),
                "both": meteor_cal.p_tx_dbw(both["C_db"], meteor_cal.t_sys_k(t, loss), band_fraction=frac),
                **{
                    k: meteor_cal.p_tx_dbw(v["C_db"], meteor_cal.t_sys_k(t, loss), band_fraction=frac)
                    for k, v in fits.items()
                },
            }
            for t in (300, 500, 1000)
        },
        "passes": [
            {
                "name": p.name,
                "aos_utc": _utc(p.p.aos),
                "max_el_deg": p.p.max_el_deg,
                "rows": len(p.t),
                "saturated_rows": int(p.saturated.sum()),
                "peak_snr_db": float(10 * np.log10(max(float(p.snr_lin.max()), 1e-3))),
            }
            for p in pds
        ],
    }
    _write_json(out_dir / "meteor_cal.json", mcal)
    prof["meteor_cal"] = {
        "file": "meteor_cal.json",
        "C_db": both["C_db"],
        "residual_rms_db": both["residual_rms_db"],
        "p_tx_dbw_at_t_ant_500": mcal["p_tx_dbw_for_t_ant"]["500"]["both"],
    }

    log("ORBCOMM reference; sample bank")
    bank = Bank(out_dir / "bank")
    orb = _orbcomm(rec, ann, rows, f, els, st, bank)
    if orb.get("channels"):
        f_orb = float(np.mean([c["channel_centre_hz"] for c in orb["channels"]]))
        orb["level"] = orbcomm.fit_level(
            orb["channels"],
            rec.start_unix + float(orb["segment_s"][0]),
            els[orb["satellite"]["norad"]],
            st,
            -saw[min(saw, key=lambda k: abs(k - f_orb))],
        )
    _write_json(out_dir / "orbcomm_ref.json", orb)
    prof["orbcomm_ref"] = {
        "file": "orbcomm_ref.json",
        "satellite": orb.get("satellite"),
        "alpha": [c["pulse"]["alpha"] for c in orb.get("channels", [])],
        "symbol_rate_hz": [c["symbol_rate_hz"] for c in orb.get("channels", [])],
        "doppler_residual_rms_hz": [c["doppler"]["residual_rms_hz"] for c in orb.get("channels", [])],
        "link_const_dbhz": orb["level"]["link_const_dbhz"] if "level" in orb else None,
    }
    _bank_bursts(rec, bank, starts, power, chans, vdl_active, rows, f)
    bank.write_index()
    prof["bank"] = {"dir": "bank", "clips": len(bank.entries), "committed": False}

    # Event list: satellite-band activity no known transmitter explains + known passes for reference
    log("event list")
    event_rows += _satband_events(raws, days, occ_days, f, els, st, bl)
    for i_, e in enumerate(sorted(event_rows, key=lambda e: (e["file"], e["row_start"], e["f_lo_hz"]))):
        e["id"] = f"e{i_:06d}"
    labels.write_events(work_dir / "events.csv", sorted(event_rows, key=lambda e: e["id"]))
    counts: dict[str, int] = {}
    for e in event_rows:
        counts[e["auto_label"]] = counts.get(e["auto_label"], 0) + 1
    prof["events"] = {"file": str(work_dir / "events.csv"), "by_auto_label": dict(sorted(counts.items()))}

    _write_json(out_dir / "profile.json", prof)
    log(f"profile written to {out_dir}")
    return prof


def _row_event(r: RawWaterfall, s: int, e: int, f_lo: float, f_hi: float, label: str, peak: float) -> dict[str, Any]:
    e = min(e, r.n_rows - 1)
    return {
        "file": r.path.name,
        "row_start": int(s),
        "row_end": int(e),
        "utc_start": _utc(float(r.row_time(s))),
        "utc_end": _utc(float(r.row_time(e))),
        "f_lo_hz": float(f_lo),
        "f_hi_hz": float(f_hi),
        "auto_label": label,
        "peak_db": round(float(peak), 2),
    }


def _comb_episodes(cs: dict[str, np.ndarray], step_s: float, threshold_db: float = 0.1) -> list[tuple[int, int]]:
    on = np.nan_to_num(cs["amp_db"]) > threshold_db
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return [
        (int(s), int(e))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)
        if (e - s) * step_s >= 30
    ]


def _wav_stft(
    rec: Recording, cache: Path, chans: list[Any], excluded: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-time power of the VDL channels over the recording, up to the first occupancy exclusion."""
    vdl = [c for c in chans if c.kind == "vdl"]
    f_c = cache / f"wavstft_{rec.path.stem}_{rec.path.stat().st_size}.npz"
    if f_c.exists():
        z = np.load(f_c)
        return z["starts"], z["power"]
    t_end = min([a for a, _b in excluded if a > 0] + [rec.duration_s])
    starts, power = occupancy.stft_channel_power(rec, vdl, 0, int(t_end * rec.fs))
    np.savez(f_c, starts=starts, power=power)
    return starts, power


def _band138_mirror(occ_days: list[Any], chans: list[Any], f: np.ndarray) -> dict[str, Any]:
    """For each 138.0-138.2 MHz event (rows), the excess at its mirror frequency compared with quiet rows."""
    j = [c.kind for c in chans].index("band138")
    from assist.profile.cache import centred_lut

    lut = centred_lut()
    ratios, mirrors = [], []
    for r, exc, peak, base in occ_days:
        ev = occupancy.events_from_series(exc[:, j], 6.0, 3.0)
        for s, e in ev[:: max(1, len(ev) // 100)]:
            k = int(peak[s + int(np.argmax(exc[s:e, j])), j])
            km = wf.FFT_SIZE - k
            db = lut[np.asarray(r.pixels[s:e])]
            h = min(s // 3600, len(base) - 1)
            ex_ev = float((db[:, k] - base[h, k]).max())
            ex_m = float((db[:, max(0, km - 2) : km + 3] - base[h, max(0, km - 2) : km + 3]).max())
            ratios.append(ex_ev)
            mirrors.append(ex_m)
    m = np.array(mirrors)
    return {
        "events_tested": len(m),
        "event_excess_db_p50": float(np.median(ratios)) if ratios else None,
        "mirror_excess_db_p50_p90": [float(np.median(m)), float(np.percentile(m, 90))] if len(m) else [],
        "mirror_detected_fraction_gt_3db": float((m > 3).mean()) if len(m) else None,
        "note": "mirror band 135.8-136.0 MHz is near the waterfall floor (SAW notch); only strong mirrors are visible",
    }


def _orbcomm(
    rec: Recording, ann: list[Any], rows: np.ndarray, f: np.ndarray, els: dict[int, dict[str, str]], st, bank: Bank
) -> dict[str, Any]:
    segs = [a for a in ann if a.file == rec.path.name and a.label.startswith("orbcomm")]
    if not segs:
        return {"note": "no orbcomm annotation"}
    a = segs[0]
    n0, n1 = int(a.from_s * rec.fs), min(rec.n_samples, int(a.to_s * rec.fs))
    t0 = rec.start_unix + n0 / rec.fs
    # Channels: peaks of the segment's median spectrum in 137.2-137.85 MHz, > 6 dB over the band median
    t_r0 = (wf.CAL_SAMPLES + np.arange(len(rows)) * wf.ROW_SAMPLES) / rec.fs
    seg = rows[(t_r0 >= a.from_s) & (t_r0 + wf.ROW_SECONDS <= a.to_s)]
    sp = np.median(seg, axis=0)
    band = (f > 137.2e6) & (f < 137.85e6)
    exc = sp - np.median(sp[band])
    hot = np.flatnonzero(band & (exc > 6))
    clusters: list[list[int]] = []
    for k in hot:
        if clusters and f[k] - f[clusters[-1][-1]] <= 10e3:
            clusters[-1].append(int(k))
        else:
            clusters.append([int(k)])
    centres = [float(np.round(f[cl[int(np.argmax(exc[cl]))]] / 100) * 100) for cl in clusters]
    cands = []
    for norad, row in els.items():
        if row["OBJECT_NAME"].startswith("ORBCOMM") and find_passes(row, st, t0, rec.start_unix + n1 / rec.fs):
            cands.append(norad)
    xs = {c: orbcomm.channelize_segment(rec, n0, n1, c) for c in centres}
    rms_per_cand: list[list[float]] = []
    for norad in cands:
        res = []
        for c, x in xs.items():
            _tr, d = orbcomm.doppler_fit_for(x, t0, c, els[norad], st)
            res.append(float(d.get("residual_rms_hz", float("inf"))))
        rms_per_cand.append(res)
    ident = [
        {"norad": n_, "name": els[n_]["OBJECT_NAME"], "residual_rms_hz": r_}
        for n_, r_ in zip(cands, rms_per_cand, strict=True)
    ]
    best_norad = cands[int(np.argmin([np.mean(r_) for r_ in rms_per_cand]))]
    chans = []
    for c, x in xs.items():
        r = orbcomm.analyse(x, t0, c, els[best_norad], st)
        chans.append(r)
        bank.add(
            rec,
            f"orbcomm_{int(r['doppler']['rest_freq_hz'])}",
            "orbcomm_stx",
            a.from_s,
            min(a.to_s, rec.duration_s),
            c,
            norad=best_norad,
            rest_freq_hz=r["doppler"]["rest_freq_hz"],
        )
    gw = (f > 137.55e6) & (f < 137.57e6)
    return {
        "segment_s": [a.from_s, a.to_s],
        "candidates": ident,
        "satellite": {"norad": best_norad, "name": els[best_norad]["OBJECT_NAME"]},
        "elements_epoch": els[best_norad]["EPOCH"],
        "channels": chans,
        "gateway_137_56_excess_db": float(np.max(sp[gw] - np.median(sp[band]))),
    }


def _bank_bursts(rec: Recording, bank: Bank, starts, power, chans, vdl_active, rows, f) -> None:
    vdl = [c for c in chans if c.kind == "vdl"]
    frame_s = 1024 / rec.fs
    for n in vdl_active:
        j = [c.name for c in vdl].index(n)
        c = vdl[j]
        for k, (a, b) in enumerate(strongest_bursts(starts / rec.fs, power[:, j], frame_s, 10)):
            bank.add(rec, f"{n}_{k:02d}", "vdl", a, b, c.centre_hz)
    # 138 MHz carriers: the strongest rows of the recording around 138.0-138.2 MHz
    band = (f > 138.0e6) & (f < 138.2e6)
    ex = rows[:, band].max(axis=1) - np.median(rows[:, band], axis=1)
    for k, i in enumerate(np.argsort(ex)[::-1][:3]):
        if ex[i] < 6:
            break
        kk = np.flatnonzero(band)[int(np.argmax(rows[i, band]))]
        t0 = (wf.CAL_SAMPLES + i * wf.ROW_SAMPLES) / rec.fs
        bank.add(rec, f"carrier138_{k}", "carrier_138", t0, t0 + wf.ROW_SECONDS, float(f[kk]))
    bank.add(rec, "pulses_136.200", "unknown", 600.0, 602.0, 136.2e6)


def _satband_events(raws, days, occ_days, f, els, st, birdie_list) -> list[dict[str, Any]]:
    fm = f / 1e6
    out = []
    for r, d, (_r, _e, _p, base) in zip(raws, days, occ_days, strict=True):
        n_min = len(d["minute_max_px"])
        t_min = np.asarray(r.row_time(np.arange(n_min) * 60 + 30), dtype=np.float64)
        known = np.zeros((n_min, len(f)), bool)
        for c in orbcomm.CHANNELS_SEEN_MHZ:
            known[:, np.abs(fm - c) < (0.025 if c == 137.56 else 0.012)] = True  # gateway bursts are ~+-23 kHz
        for b in birdie_list:
            known[:, np.abs(f - b["freq_hz"]) < 1e3] = True
        known[:, np.abs(fm - 137.0) < 0.006] = True
        for norad in METEORS:
            for p in find_passes(els[norad], st, t_min[0], t_min[-1]):
                m = (t_min >= p.aos - 120) & (t_min <= p.los + 120)
                known[np.ix_(m, np.abs(fm - 137.9) < 0.07)] = True
        for m0, m1, b0, b1, pk in labels.satband_events(d["minute_max_px"], base, f, known):
            centre = (f[b0] + f[b1 - 1]) / 2
            guess = "lora" if any(lo <= centre <= hi for lo, hi in LORA_CHANNELS_HZ) else "unknown_sat"
            out.append(_row_event(r, m0 * 60, m1 * 60, float(f[b0]), float(f[b1 - 1]), guess, pk))
        for norad in METEORS:
            for p in find_passes(els[norad], st, t_min[0], t_min[-1]):
                if p.max_el_deg > 5:
                    out.append(_row_event(r, r.row_at(p.aos), r.row_at(p.los), 137.85e6, 137.95e6, "meteor", 0.0))
    return out
