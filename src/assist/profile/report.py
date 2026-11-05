"""P13: profile report (plots + Markdown summary), from a profile directory and the build's intermediates."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from assist import waterfall as wf
from assist.profile import humps, noise
from assist.profile.adc import hist_moments

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _load(profile_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((profile_dir / name).read_text())


def write_report(profile_dir: Path, out: Path, work: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    prof = _load(profile_dir, "profile.json")
    birds = _load(profile_dir, "birdies.json")["birdies"]
    orb = _load(profile_dir, "orbcomm_ref.json")
    met = _load(profile_dir, "meteor_cal.json")
    psd = np.load(profile_dir / "noise_psd.npy")
    hourly = np.load(profile_dir / "noise_hourly.npy")
    cache = work / "cache"
    f = wf.bin_freqs(prof["fc_hz"]) / 1e6
    rec_in = prof["inputs"]["recording"]
    wav_key = f"{Path(rec_in['file']).stem}_{rec_in['bytes']}"
    days = {
        w["file"]: dict(np.load(cache / f"day_{Path(w['file']).stem}_{w['bytes']}.npz"))
        for w in prof["inputs"]["waterfalls"]
    }
    figs: list[tuple[str, str]] = []

    def save(fig: Figure, name: str, caption: str) -> None:
        fig.savefig(out / name, bbox_inches="tight", dpi=90)
        plt.close(fig)
        figs.append((name, caption))

    # Noise PSD
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))
    for name, d in days.items():
        ax[0].plot(f, noise.normalise(np.median(d["minute_mean_db"], axis=0), f), lw=0.5, label=name)
    z = np.load(cache / f"wavrows_{wav_key}.npz")
    ax[0].plot(f, noise.normalise(np.median(z["rows"], axis=0), f), lw=0.5, label="recording (all rows)")
    ax[0].plot(f, psd, "k", lw=1.2, label="profile noise_psd")
    for b in birds:
        ax[0].axvline(b["freq_hz"] / 1e6, color="r", lw=0.4, alpha=0.5)
    ax[0].set_ylim(-14, 6)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Noise shape (0 dB = median 136.2-137.8 MHz); red: birdies")
    k = np.abs(f - 137.0) < 0.02
    for name, d in days.items():
        ax[1].plot(f[k], noise.normalise(np.median(d["minute_mean_db"], axis=0), f)[k], ".-", lw=0.5, label=name)
    ax[1].plot(f[k], noise.normalise(np.median(z["rows"], axis=0), f)[k], ".-", lw=0.5, label="recording")
    ax[1].set_title("Around DC (137.000 MHz)")
    ax[1].legend(fontsize=7)
    save(fig, "noise_psd.png", "Noise shape from the two days and the recording; the profile PSD.")

    # Diurnal + regimes
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    for name, d in days.items():
        lvl = d["row_level_db"]
        ax[0].plot(np.arange(len(lvl)) / 3600, lvl - np.median(lvl), lw=0.3, label=name)
    ax[0].set_xlabel("hours since file start")
    ax[0].set_ylabel("band level, dB rel. median")
    ax[0].legend(fontsize=7)
    ax[1].bar(np.arange(24), hourly)
    ax[1].set_xlabel("UTC hour")
    ax[1].set_ylabel("dB")
    ax[1].set_title("noise_hourly (mean of the days)")
    save(fig, "noise_hourly.png", "Band level per row and the hourly curve; note the step-like regimes.")

    # ADC
    a = np.load(next(cache.glob(f"adc_{wav_key}.npz")))
    hi, hq = a["hist_i"].sum(axis=0), a["hist_q"].sum(axis=0)
    mi = hist_moments(a["hist_i"])
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    v = np.arange(256)
    ax[0].semilogy(v, hi / hi.sum(), lw=0.8, label="I")
    ax[0].semilogy(v, hq / hq.sum(), lw=0.8, label="Q")
    sig = prof["adc"]["noise_sigma_mad_counts"]
    g = np.exp(-0.5 * ((v - 127.5) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    ax[0].semilogy(v, g, "k--", lw=0.8, label=f"Gaussian sigma {sig:.1f}")
    ax[0].set_ylim(1e-8, 1e-1)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Sample value histogram (whole recording)")
    ax[1].plot(a["t0"], mi["sigma_mad"], lw=0.7, label="sigma (MAD), I")
    ax[1].plot(a["t0"], mi["rms"], lw=0.7, label="RMS, I")
    ax2 = ax[1].twinx()
    ax2.plot(a["t0"], 100 * mi["clip"], "r", lw=0.5)
    ax2.set_ylabel("clipped %", color="r")
    ax[1].legend(fontsize=7)
    ax[1].set_xlabel("s")
    save(fig, "adc.png", "ADC: heavy tails and pile-up at 1/254 (clipping) from strong bursts.")

    # Occupancy
    occ = prof["occupancy"]
    fig, ax = plt.subplots(1, 1, figsize=(15, 4.5))
    for n, v_ in occ["vdl"].items():
        ax.plot(range(24), v_["events_per_hour_utc"], "o-", label=n)
    ax.plot(range(24), occ["am"]["events_per_hour_utc"], "s-", label="am (all channels, <= 60 s)")
    if occ.get("band138"):
        ax.plot(range(24), occ["band138"]["events_per_hour_utc"], "^-", label="138.0-138.2 MHz")
    ax.set_yscale("log")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("events per hour (1 s rows)")
    ax.legend(fontsize=7)
    save(fig, "occupancy.png", "Hourly event rates from the waterfall days.")

    # ORBCOMM
    if orb.get("channels"):
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
        for c in orb["channels"]:
            s = c["series"]
            good = np.array(s["peak_db"]) > 15
            t = np.array(s["t_s"])
            ax[0].plot(
                t[good],
                np.array(s["residual_hz"])[good],
                ".",
                ms=3,
                label=f"{c['doppler']['rest_freq_hz'] / 1e6:.6f} MHz",
            )
            ax[1].plot(t, s["cn0_dbhz"], lw=0.8)
            ax[2].plot(c["psd"]["f_hz"], c["psd"]["psd_db"], lw=0.6)
            ax[2].plot(c["psd"]["f_hz"], c["psd"]["model_db"], "k--", lw=0.8)
        ax[0].set_title("Doppler residual (Hz) vs s into segment")
        ax[0].legend(fontsize=7)
        ax[1].set_title("C/N0 (dB-Hz)")
        ax[2].set_title("PSD (dB over noise) and RC fit")
        save(fig, "orbcomm.png", f"ORBCOMM reference: {orb['satellite']['name']}.")

    # METEOR
    rt = met["residual_tables"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    eb = np.array(rt["el_bins"])
    ax[0].errorbar(eb[:, 0] + 2.5, eb[:, 2], yerr=eb[:, 3], fmt="o")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("elevation (deg)")
    ax[0].set_ylabel("measured - model (dB)")
    ab = np.array(rt["az_bins"])
    ax[1].errorbar(ab[:, 0] + 15, ab[:, 2], yerr=ab[:, 3], fmt="o")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("azimuth (deg)")
    ps = met["passes"]
    ax[2].plot([p["max_el_deg"] for p in ps], [p["peak_snr_db"] for p in ps], "o")
    ax[2].set_xlabel("max elevation")
    ax[2].set_ylabel("peak SNR (dB, 100 kHz)")
    save(fig, "meteor.png", "METEOR calibration residuals (QFH at both ends, FSPL).")

    # Humps
    fcol = wf.bin_freqs(prof["fc_hz"]).reshape(-1, humps.COL_BINS).mean(axis=1) / 1e6
    step_s = humps.STEP_ROWS * wf.ROW_SECONDS
    cubes = sorted(cache.glob("humps_*.npy"))
    if cubes:
        fig, ax = plt.subplots(len(cubes), 1, figsize=(15, 3 * len(cubes)), squeeze=False)
        for axi, cpath in zip(ax[:, 0], cubes, strict=True):
            cs = humps.comb_series(np.load(cpath), fcol)
            t = np.arange(len(cs["amp_db"])) * step_s / 3600
            axi.plot(t, cs["amp_db"], lw=0.4)
            axi2 = axi.twinx()
            axi2.plot(t, cs["spacing_hz"] / 1e3, "r.", ms=0.6)
            axi2.set_ylabel("spacing kHz", color="r")
            axi.set_title(f"{cpath.stem}: comb ripple amplitude (dB)")
            axi.set_xlabel("hours")
        save(fig, "humps.png", "Hump comb amplitude and spacing per 10 s step.")

    md = _markdown(prof, birds, orb, met, figs)
    path = out / "report.md"
    path.write_text(md)
    return path


def _f(x: object, nd: int = 2) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, list):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    return str(x)


def _markdown(
    prof: dict[str, Any], birds: list[Any], orb: dict[str, Any], met: dict[str, Any], figs: list[tuple[str, str]]
) -> str:
    a, n, o = prof["adc"], prof["noise"], prof["occupancy"]
    rec, dc = prof["recording"], n["dc_spike"]
    clip = f"{_f(100 * a['clip_fraction_median'])} % median, {_f(100 * a['clip_fraction_max_1s'])} % max per second"
    iq = f"gain ratio {_f(a['iq_gain_ratio'], 5)}, phase {_f(a['iq_phase_deg'], 4)} deg"
    shape = f"{_f(n['recording_vs_days_shape_rms_db'])} dB (day vs day {_f(n['days_shape_rms_db'])} dB)"
    fit = met["fit_both"]
    vdl = []
    for k, v in o["vdl"].items():
        fine = v["recording_fine_time"]["duration_s"].get("median", 0) * 1e3
        vdl.append(
            f"- {k}: {_f(v['events_per_day'])} events/day; row-duration median "
            f"{_f(v['duration_s'].get('median'), 1)} s; fine-time bursts median {_f(fine, 1)} ms"
        )
    orbl = []
    for c in orb.get("channels", []):
        d, pu = c["doppler"], c["pulse"]
        orbl += [
            f"- {d['rest_freq_hz'] / 1e6:.6f} MHz: Doppler residual {_f(d['residual_rms_hz'])} Hz RMS; best time "
            f"offset {_f(d['best_time_offset_s'])} s -> {_f(d['residual_rms_at_best_offset_hz'])} Hz",
            f"  rs {_f(c['symbol_rate_hz'], 3)} Hz; alpha {_f(pu['alpha'], 3)} +- {_f(pu['alpha_stderr'], 3)}; "
            f"C/N0 p10/50/90 {_f(c['cn0_dbhz_p10_p50_p90'], 1)} dB-Hz",
        ]
    b138 = _f(o["band138"]["events_per_day"], 1) if o.get("band138") else "n/a"
    lines = [
        "# Profile report: vhf137",
        "",
        f"Recording: {prof['inputs']['recording']['file']} ({_f(rec['duration_s'], 1)} s from {rec['start_utc']}).",
        f"Waterfalls: {', '.join(w['file'] for w in prof['waterfalls'])}.",
        "",
        "## ADC (P2)",
        f"- noise sigma (MAD) {_f(a['noise_sigma_mad_counts'])} counts; RMS {_f(a['rms_counts'])} counts",
        f"- clipped (1/254 rails): {clip}",
        f"- DC {_f(a['dc_i_counts'], 3)} / {_f(a['dc_q_counts'], 3)} counts; IQ {iq}",
        f"- image from noise statistics {_f(a['image_rejection_from_stats_db'], 1)} dB",
        "",
        "## Noise (P3)",
        f"- recording vs days shape RMS {shape}",
        f"- DC spike (days) {_f(dc['centre_bin_excess_db'])} dB in the centre bin",
        f"- |DC| ~ {_f(dc['amplitude_counts_est'], 3)} counts (from the spike and the noise sigma)",
        f"- hourly (UTC) {_f(n['hourly_utc_db'], 1)}",
        "",
        f"## Birdies (P6): {len(birds)}",
        "",
        "| MHz | excess dB | width bins |",
        "|---|---|---|",
        *[f"| {b['freq_hz'] / 1e6:.4f} | {_f(b['excess_db'])} | {b['width_bins']} |" for b in birds],
        "",
        "## Occupancy (P7)",
        *vdl,
        f"- AM-grid short events: {_f(o['am']['events_per_day'], 1)}/day; top {o['am']['top_channels'][:5]}",
        f"- 138.0-138.2 MHz: {b138}/day",
        "",
        "## ORBCOMM reference (P2b)",
        f"- satellite {orb.get('satellite')}",
        *orbl,
        "",
        "## METEOR calibration (P10)",
        f"- C = {_f(fit['C_db'])} dB over {fit['rows']} rows; residual {_f(fit['residual_rms_db'])} dB RMS",
        f"- P_tx at T_ant 500 K: {_f(met['p_tx_dbw_for_t_ant']['500']['both'], 1)} dBW (degenerate with T_sys)",
        "",
        "## Figures",
        *[f"![{c}]({n_})\n\n{c}\n" for n_, c in figs],
    ]
    return "\n".join(lines) + "\n"
