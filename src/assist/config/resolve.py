"""`resolve`: turn a scenario into a self-contained canonical document.

After loading (includes, --set, band defaults, presets), drawing random expressions and validating, resolve fills
in everything that depends on files or draws:
- clock start (`now` -> the current UTC second), station position (file or fallback);
- the element set of every catalog orbit, embedded as an OMM dict (nearest the clock start, or `orbit.epoch`);
- synthetic passes: absolute AOS, direction/side (keyed draws if absent) and the solved orbit;
- keyed defaults: on/off phase, AM phase; channels (names from the band's channel table, e.g. ORBCOMM S-1 .. S-12)
  turned into frequencies, an emitter's `channel` into its freq_hz; frequencies quantised to 1 mHz;
- levels as p_tx_dbw (from p_tx_dbw, cn0_zenith_dbhz or a calibration); ground and local transmitters as
  rx_dbm (from cn0_dbhz or rx_dbm);
- package version and content hashes of the band and profile.

Resolving a resolved document returns it unchanged. The scenario hash is the sha256 of the canonical JSON (sorted
keys, compact separators).
"""

import datetime as dt
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assist import __version__, plugins
from assist.config import schema as S
from assist.config.loader import ConfigError, band_dir, load_band, load_scenario
from assist.config.randexpr import draw_all
from assist.emitters import link
from assist.orbits.elements import (
    ELEMENTS_DIR,
    ElementSet,
    ElementStore,
    epoch_unix,
    latest_snapshots,
    staleness_warning,
)
from assist.orbits.synthetic import RE_KM, PassRequest, solve_pass
from assist.platforms.base import platform_class
from assist.rng import KeyedRng
from assist.station import DEFAULT_FILE, Station, load_station
from assist.timebase import format_iso, parse_iso, parse_time

MU_KM3_S2 = 398600.8
METEOR_BAND_HZ = 100e3  # the METEOR calibration measured SNR in 137.9 +- 50 kHz
METEOR_AUTO = "meteor_auto"  # the per-satellite METEOR calibration of the source's NORAD id, else `meteor`


@dataclass
class Resolved:
    scenario: S.Scenario
    warnings: list[str] = field(default_factory=list)
    clock_now: bool = False  # the scenario's clock said `start: now` (a live stream starts at the wall clock)

    @property
    def canonical(self) -> str:
        return canonical_json(self.scenario)

    @property
    def hash(self) -> str:
        """sha256 of the canonical JSON without the `sinks`: the hash names what is rendered, not where it goes, so
        changing a sink setting (`--set 'sinks[wav].rf64=true'`) leaves it alone."""
        doc = json.loads(self.canonical)
        doc["sinks"] = []
        text = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def pretty(self) -> str:
        return json.dumps(json.loads(self.canonical), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def canonical_json(sc: S.Scenario) -> str:
    """The resolved scenario as canonical JSON (the scenario hash is its sha256). An empty `plugins` list is left
    out, so scenarios that use only core modules keep the hash they had before plugins existed."""
    doc = sc.model_dump(mode="json")
    if doc["plugins"]:
        doc["plugins"] = [{k: v for k, v in p.items() if v is not None} for p in doc["plugins"]]
    else:
        del doc["plugins"]
    for src in doc["sources"]:  # likewise the core platforms, which have no parameters
        if not src["platform_params"]:
            del src["platform_params"]
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dir_hash(path: Path, exclude: tuple[str, ...] = ()) -> str:
    """sha256 over the relative paths and contents of every file below path (sorted), skipping `exclude` dirs."""
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = p.relative_to(path).as_posix()
        if rel.split("/")[0] in exclude or "__pycache__" in rel:
            continue
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def calibrations(profile: str | None, band: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Calibrated link constants (dB-Hz) by name, from the profile when present, else from the band defaults."""
    if profile:
        pdir = Path(profile)
        out: dict[str, float] = {}
        mc = json.loads((pdir / "meteor_cal.json").read_text())
        frac = float(mc["band_power_fraction"])
        for name, fit in mc["fit_per_satellite"].items():
            key = name.lower().replace("-", "_").replace(" ", "_")
            out[key] = link.link_const_from_snr_c(float(fit["C_db"]), METEOR_BAND_HZ, frac)
        out["meteor"] = link.link_const_from_snr_c(float(mc["fit_both"]["C_db"]), METEOR_BAND_HZ, frac)
        orb = json.loads((pdir / "orbcomm_ref.json").read_text())
        if "level" in orb:
            out["orbcomm"] = float(orb["level"]["link_const_dbhz"])
        return out, f"profile {profile}"
    cal = band.get("calibrations", {})
    return {k: float(v["link_const_dbhz"]) for k, v in cal.items()}, f"band {band.get('name', '')}"


def _station(st: S.StationConfig, base: Path) -> S.StationConfig:
    fields = st.model_fields_set
    path: str | None
    if "file" in fields:
        path = st.file
    elif {"lat", "lon"} & fields:
        path = None
    else:
        path = DEFAULT_FILE
    if path is not None:
        p = Path(path) if Path(path).is_absolute() else base / path
        if p.is_file():
            s = load_station(p)
            tz = s.tz if "tz" not in fields else st.tz
            return st.model_copy(
                update={"file": None, "lat": s.lat_deg, "lon": s.lon_deg, "alt_m": s.alt_m, "tz": tz, "source": path}
            )
    source = st.source or ("config" if {"lat", "lon"} & fields else "fallback")
    return st.model_copy(update={"file": None, "source": source})


def satellite_height_m(orbit: S.CatalogOrbit | S.KeplerOrbit | S.SyntheticPassOrbit, station_alt_m: float) -> float:
    """Height of the orbit above the station, for cn0_zenith_dbhz (mean semi-major axis minus Earth radius)."""
    if isinstance(orbit, S.CatalogOrbit):
        assert orbit.omm is not None
        n = float(orbit.omm["MEAN_MOTION"]) * 2 * math.pi / 86400
        a_km = math.cbrt(MU_KM3_S2 / n**2)
    elif isinstance(orbit, S.KeplerOrbit):
        k = orbit.kepler
        a_km = float(k.a_km) if k.a_km is not None else RE_KM + float(k.alt_km or 0.0)
    else:
        a_km = RE_KM + float(orbit.synthetic_pass.alt_km)
    return (a_km - RE_KM) * 1000 - station_alt_m


def pattern_of(p: Any) -> link.Pattern:
    if isinstance(p, S.PolyPattern):
        return link.Pattern("poly", coeffs_db=tuple(p.coeffs_db))
    if isinstance(p, S.CardioidPattern):
        return link.Pattern("cardioid", n=p.n, peak_dbi=p.peak_dbi)
    return link.Pattern("isotropic", gain_dbi=p.gain_dbi)


def t_sys_of(fe: S.Frontend) -> float:
    return link.t_sys_k(fe.t_ant_k, fe.saw_loss_db, fe.lna.nf_db, fe.lna.gain_db, fe.tuner_nf_db)


class _Ctx:
    def __init__(self, sc: S.Scenario, base: Path, band: dict[str, Any]) -> None:
        self.sc, self.base, self.band = sc, base, band
        self.rng = KeyedRng(sc.seed)
        self.t0 = parse_iso(sc.clock.start)
        self.warnings: list[str] = []
        self._store: ElementStore | None = None
        self._cal: tuple[dict[str, float], str] | None = None

    def store(self) -> ElementStore:
        if self._store is None:
            files = [self.base / f for f in self.sc.elements.files]
            missing = [str(f) for f in files if not f.is_file()]
            if missing:
                raise ConfigError(f"element files not found: {missing}")
            self._store = ElementStore.load(files)
        return self._store

    def cal(self) -> tuple[dict[str, float], str]:
        if self._cal is None:
            prof = str(self.base / self.sc.profile) if self.sc.profile else None
            table, _ = calibrations(prof, self.band)
            # Name the source as the scenario does (not the absolute path), so the resolved document, and with it
            # the scenario hash, is the same in every checkout.
            where = f"profile {self.sc.profile}" if self.sc.profile else f"band {self.band.get('name', '')}"
            self._cal = (table, where)
        return self._cal

    def uniform(self, sid: str, path: str, lo: float, hi: float) -> float:
        return float(self.rng.stream("resolve", sid, path).uniform(lo, hi))

    def choice(self, sid: str, path: str, items: list[str]) -> str:
        return items[int(self.rng.stream("resolve", sid, path).integers(len(items)))]


def _concrete_elements(cx: _Ctx) -> S.Scenario:
    """Turn `elements.latest` / `latest_existing` into the file list they stand for (downloading for `latest`).

    The resolved scenario lists the files and sets both flags to false, so catalog tools and a re-resolve of the
    resolved document read the same files and never download again.
    """
    el = cx.sc.elements
    if not (el.latest or el.latest_existing):
        return cx.sc
    try:
        paths, warns = latest_snapshots(cx.base / ELEMENTS_DIR, download=el.latest)
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e
    cx.warnings += warns
    files = [p.relative_to(cx.base).as_posix() for p in paths]
    files += [f for f in el.files if f not in files]
    return cx.sc.model_copy(update={"elements": S.Elements(files=files)})


def synthetic_norad(source_id: str) -> int:
    return 90000 + int.from_bytes(hashlib.blake2b(source_id.encode(), digest_size=8).digest(), "little") % 10000


def _orbit(cx: _Ctx, src: S.Source, station: Station) -> tuple[Any, bool]:
    o = src.orbit
    if isinstance(o, S.CatalogOrbit):
        if o.omm is None:
            t_sel = parse_iso(o.epoch) if o.epoch else cx.t0
            try:
                es = cx.store().nearest(o.norad, t_sel)
            except KeyError as e:
                where = ", ".join(cx.sc.elements.files) or "none; set elements.files, latest or latest_existing"
                raise ConfigError(f"source {src.id}: NORAD {o.norad} is in none of the element files ({where})") from e
            o = o.model_copy(update={"omm": es.omm})
        w = staleness_warning(_set_of(o), cx.t0)
        if w:
            cx.warnings.append(f"source {src.id}: {w}")
        return o, True
    if isinstance(o, S.KeplerOrbit):
        return o, False
    if o is None:
        raise ConfigError(f"source {src.id}: a satellite needs an orbit")
    sp = o.synthetic_pass
    aos = parse_time(sp.aos, cx.t0)
    direction = sp.direction or cx.choice(src.id, "orbit.synthetic_pass.direction", ["asc", "desc"])
    side = sp.side or cx.choice(src.id, "orbit.synthetic_pass.side", ["east", "west"])
    sol = sp.solution
    if sol is None:
        req = PassRequest(aos, sp.max_el_deg, direction, side, sp.alt_km, sp.inc_deg)  # ty: ignore[invalid-argument-type]
        raan, m = solve_pass(req, station, cx.sc.station.horizon_deg)
        sol = S.PassSolution(raan_deg=raan, m_deg=m)
    sp = sp.model_copy(update={"aos": format_iso(aos), "direction": direction, "side": side, "solution": sol})
    return S.SyntheticPassOrbit(synthetic_pass=sp), False


def _set_of(o: S.CatalogOrbit) -> ElementSet:
    assert o.omm is not None
    return ElementSet(o.norad, str(o.omm.get("OBJECT_NAME", "")), epoch_unix(o.omm), o.omm)


def _fixed_level(cx: _Ctx, src: S.Source, e: S.Emitter) -> S.Level | None:
    """Platforms with a received level: the level as rx_dbm (from cn0_dbhz against k T_sys, or rx_dbm)."""
    lv = e.level
    if lv is None or lv.resolved_from is not None:
        return lv
    if lv.cn0_dbhz is None and lv.rx_dbm is None:
        raise ConfigError(f"source {src.id}: emitter {e.id}: a {src.platform} level is cn0_dbhz or rx_dbm")
    if lv.rx_dbm is not None:
        rx, how = lv.rx_dbm - lv.extra_loss_db, "rx_dbm"
    else:
        t_sys = t_sys_of(cx.sc.frontend)
        assert lv.cn0_dbhz is not None
        rx = lv.cn0_dbhz - lv.extra_loss_db + 30 + link.kt_dbw_hz(t_sys)
        how = f"cn0_dbhz {lv.cn0_dbhz} at T_sys {t_sys:.1f} K"
    if lv.extra_loss_db:
        how += f", extra loss {lv.extra_loss_db} dB"
    return S.Level(rx_dbm=rx, resolved_from=how)


def _level(cx: _Ctx, src: S.Source, orbit: Any, e: S.Emitter, station_alt_m: float) -> S.Level | None:
    if platform_class(src.platform).level_mode == "received":
        return _fixed_level(cx, src, e)
    lv = e.lvl
    if lv.resolved_from is not None:
        if lv.p_tx_dbw is None:
            raise ConfigError(f"source {src.id}: emitter {e.id}: a satellite's resolved level is p_tx_dbw")
        return lv
    if lv.cn0_dbhz is not None or lv.rx_dbm is not None:
        raise ConfigError(
            f"source {src.id}: emitter {e.id}: cn0_dbhz / rx_dbm are for ground and local sources; "
            "a satellite takes p_tx_dbw, cn0_zenith_dbhz or cal"
        )
    t_sys = t_sys_of(cx.sc.frontend)
    if lv.p_tx_dbw is not None:
        p, how = lv.p_tx_dbw, "p_tx_dbw"
    elif lv.cn0_zenith_dbhz is not None:
        h = satellite_height_m(orbit, station_alt_m)
        p = link.p_tx_for_cn0_zenith(
            lv.cn0_zenith_dbhz,
            h,
            e.freqs_hz[0],
            pattern_of(e.pattern),
            pattern_of(cx.sc.antenna.pattern),
            t_sys,
            lv.extra_loss_db,
        )
        how = f"cn0_zenith_dbhz {lv.cn0_zenith_dbhz} at {h / 1000:.1f} km, T_sys {t_sys:.1f} K"
        if len(e.freqs_hz) > 1:
            how += f" (at {e.freqs_hz[0] / 1e6:.4f} MHz, the first frequency)"
    else:
        table, where = cx.cal()
        cal = str(lv.cal)
        if cal == METEOR_AUTO:
            norad = orbit.norad if isinstance(orbit, S.CatalogOrbit) else None
            by_norad = {int(v["norad"]): k for k, v in cx.band.get("calibrations", {}).items() if "norad" in v}
            cal = by_norad.get(norad, "meteor") if norad is not None else "meteor"
            lv = lv.model_copy(update={"cal": cal})
        if lv.cal not in table:
            raise ConfigError(f"source {src.id}: unknown calibration {lv.cal!r} in {where} (known: {sorted(table)})")
        k = table[str(lv.cal)]
        p = link.p_tx_from_link_const(k, t_sys)
        how = f"cal {lv.cal} from {where}: link constant {k:.3f} dB-Hz, T_sys {t_sys:.1f} K"
        if e.lvl.cal == METEOR_AUTO:
            how = f"cal {METEOR_AUTO} -> {how}"
    return S.Level(p_tx_dbw=p, extra_loss_db=lv.extra_loss_db, resolved_from=how)


def _mhz_round(f: float) -> float:
    return round(f * 1000) / 1000


def channel_hz(band: dict[str, Any], name: str, sid: str) -> float:
    table = band.get("channels", {})
    if name not in table:
        raise ConfigError(f"source {sid}: unknown channel {name!r} (band channels: {', '.join(table)})")
    return float(table[name])


def _channels(cx: _Ctx, src: S.Source) -> list[float | str] | None:
    if src.channels is None:
        return None
    return [_mhz_round(channel_hz(cx.band, c, src.id) if isinstance(c, str) else float(c)) for c in src.channels]


def _emitter(
    cx: _Ctx, src: S.Source, orbit: Any, e: S.Emitter, station_alt_m: float, channels: list[float | str] | None
) -> S.Emitter:
    if e.channel is not None:
        ch = e.channel
        f = channel_hz(cx.band, ch, src.id) if isinstance(ch, str) else float((channels or [])[ch])
        e = e.model_copy(update={"freq_hz": f, "channel": None})
    freq = [_mhz_round(f) for f in e.freqs_hz]
    e = e.model_copy(update={"freq_hz": freq if isinstance(e.freq_hz, list) else freq[0]})
    upd: dict[str, Any] = {
        "offset_hz": round(e.offset_hz * 1000) / 1000,
        "level": _level(cx, src, orbit, e, station_alt_m),
    }
    sch = e.schedule
    if isinstance(sch, S.OnOffSchedule) and sch.phase_s is None:
        ph = cx.uniform(src.id, f"emitters[{e.id}].schedule.phase_s", 0.0, sch.on_s + sch.off_s)
        upd["schedule"] = sch.model_copy(update={"phase_s": round(ph, 3)})
    if e.am is not None and e.am.phase_rad is None:
        ph = cx.uniform(src.id, f"emitters[{e.id}].am.phase_rad", 0.0, 2 * math.pi)
        upd["am"] = e.am.model_copy(update={"phase_rad": ph})
    return e.model_copy(update=upd)


def resolve_document(
    doc: dict[str, Any], base: Path, now: float | None = None, scenario_dir: Path | None = None
) -> Resolved:
    """Resolve a merged document (after load_scenario). `base` is the directory relative paths start from;
    `scenario_dir` (the scenario file's directory) is where a plugin file is looked for when it is not under `base`.

    The plugins the document lists are loaded first: their kinds must be known before the schema checks them."""
    try:
        entries = plugins.activate(doc.get("plugins"), (base, scenario_dir))
    except plugins.PluginError as e:
        raise ConfigError(str(e)) from e
    if "plugins" in doc:
        doc = {**doc, "plugins": entries}
    seed = doc.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"seed must be an integer >= 0, got {seed!r}")
    sc = S.Scenario.model_validate(draw_all(doc, KeyedRng(seed)))
    band = load_band(sc.band)
    clock_now = sc.clock.start == "now"
    if clock_now:
        t_now = math.floor(time.time() if now is None else now)
        sc = sc.model_copy(update={"clock": sc.clock.model_copy(update={"start": format_iso(t_now)})})
    sc = sc.model_copy(update={"station": _station(sc.station, base)})
    cx = _Ctx(sc, base, band)
    sc = cx.sc = _concrete_elements(cx)
    station = Station(sc.station.lat, sc.station.lon, sc.station.alt_m, sc.station.tz)
    sources = []
    for src in sc.sources:
        if platform_class(src.platform).needs_orbit:
            orbit, catalog = _orbit(cx, src, station)
        else:
            orbit, catalog = None, False
        in_cat = src.in_catalog if src.in_catalog is not None else catalog
        norad = src.catalog_norad
        if in_cat and not catalog and norad is None:
            norad = synthetic_norad(src.id)
        channels = _channels(cx, src)
        ems = [_emitter(cx, src, orbit, e, station.alt_m, channels) for e in src.emitters]
        sources.append(
            src.model_copy(
                update={
                    "orbit": orbit,
                    "in_catalog": in_cat,
                    "catalog_norad": norad,
                    "channels": channels,
                    "emitters": ems,
                }
            )
        )
    _check_norads(sources)
    prof = base / sc.profile if sc.profile else None
    if prof is not None and not prof.is_dir():
        raise ConfigError(f"profile directory {prof} not found")
    sc = sc.model_copy(
        update={
            "sources": sources,
            "assist_version": __version__,
            "band_hash": dir_hash(band_dir(sc.band)),
            "profile_hash": dir_hash(prof, exclude=("bank",)) if prof else None,
        }
    )
    # Round trip through validation, so the result is exactly what re-reading the JSON gives.
    sc = S.Scenario.model_validate(json.loads(canonical_json(sc)))
    return Resolved(sc, cx.warnings, clock_now)


def _check_norads(sources: list[S.Source]) -> None:
    """A synthetic object in the catalog must not share its NORAD id with any other catalog entry."""
    real = {s.orbit.norad: s.id for s in sources if isinstance(s.orbit, S.CatalogOrbit) and s.in_catalog}
    synth: dict[int, str] = {}
    for s in sources:
        if isinstance(s.orbit, S.CatalogOrbit) or not s.in_catalog or s.catalog_norad is None:
            continue
        other = real.get(s.catalog_norad) or synth.get(s.catalog_norad)
        if other is not None:
            raise ConfigError(f"sources {other} and {s.id} share catalog NORAD {s.catalog_norad}; set catalog_norad")
        synth[s.catalog_norad] = s.id


def resolve(
    path: str | Path,
    sets: tuple[str, ...] | list[str] = (),
    now: float | None = None,
    base: Path | None = None,
    extra_plugins: tuple[str, ...] | list[str] = (),
) -> Resolved:
    """Load and resolve a scenario file (YAML or JSON, including an already resolved one). `extra_plugins`
    (`--plugin`) are added to the scenario's `plugins:` list, so they are part of the resolved scenario."""
    doc = load_scenario(path, sets)
    if extra_plugins:
        listed = doc.get("plugins") or []
        if not isinstance(listed, list):
            raise ConfigError(f"plugins must be a list, got {listed!r}")
        doc["plugins"] = [*listed, *(plugins.from_cli(p) for p in extra_plugins)]
    return resolve_document(doc, base or Path.cwd(), now, Path(path).resolve().parent)


def utc_now() -> float:
    return dt.datetime.now(dt.UTC).timestamp()
