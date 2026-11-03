"""P2: ADC statistics of a cu8 recording.

Per second of samples: a 256-bin histogram of I and of Q, and the I/Q covariance. From these: DC (mean counts
relative to 127.5), RMS, a robust noise sigma (1.4826 * MAD, insensitive to strong bursts), and clipping. The
station's samples rail at 1 and 254 (0 and 255 are rare), so both pairs count as clipped.
IQ imbalance from noise statistics: gain ratio g = rms_Q / rms_I, phase error phi = asin(rho) with
rho = cov(I,Q) / (rms_I rms_Q). Under the model y' = y + beta * conj(y), the image rejection is
IRR = (1 + 2 g cos(phi) + g^2) / (1 - 2 g cos(phi) + g^2).
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from assist.profile.annotations import keep_mask

if TYPE_CHECKING:
    from assist.profile.wavio import Recording


def per_second_stats(rec: Recording, seconds: float = 1.0) -> dict[str, np.ndarray]:
    """Per block of `seconds`: t0 (s from file start), hist_i, hist_q (256 counts each), cov_iq (counts^2)."""
    blk = round(seconds * rec.fs)
    n_blk = rec.n_samples // blk
    out = {
        "t0": np.arange(n_blk) * blk / rec.fs,
        "hist_i": np.empty((n_blk, 256), np.int64),
        "hist_q": np.empty((n_blk, 256), np.int64),
        "cov_iq": np.empty(n_blk),
    }
    for b in range(n_blk):
        u = rec.u8(b * blk, blk).reshape(-1, 2)
        out["hist_i"][b] = np.bincount(u[:, 0], minlength=256)
        out["hist_q"][b] = np.bincount(u[:, 1], minlength=256)
        i = u[:, 0].astype(np.float32)
        q = u[:, 1].astype(np.float32)
        i -= np.float32(i.mean(dtype=np.float64))
        q -= np.float32(q.mean(dtype=np.float64))
        out["cov_iq"][b] = np.dot(i, q) / len(i)
    return out


_V = np.arange(256) - 127.5


def hist_moments(h: np.ndarray) -> dict[str, np.ndarray]:
    """From histograms (..., 256): mean, rms (about the mean), robust sigma (1.4826 MAD), clip fraction."""
    n = h.sum(axis=-1)
    mean = (h * _V).sum(axis=-1) / n
    rms = np.sqrt((h * (_V - mean[..., None]) ** 2).sum(axis=-1) / n)
    c = np.cumsum(h, axis=-1)
    med = np.argmax(c >= (n[..., None] + 1) / 2, axis=-1) - 127.5
    dev = np.abs(_V[None, :] - np.atleast_1d(med)[:, None])  # MAD via the histogram of |v - median|
    order = np.argsort(dev, axis=-1, kind="stable")
    hs = np.take_along_axis(np.atleast_2d(h), order, axis=-1)
    ds = np.take_along_axis(dev, order, axis=-1)
    k = np.argmax(np.cumsum(hs, axis=-1) >= (np.atleast_1d(n)[:, None] + 1) / 2, axis=-1)
    mad = ds[np.arange(len(k)), k].reshape(np.shape(n))
    clip = (h[..., :2].sum(axis=-1) + h[..., 254:].sum(axis=-1)) / n
    return {"mean": mean, "rms": rms, "sigma_mad": 1.4826 * mad, "clip": clip}


def image_rejection_db(gain_ratio: float, phase_rad: float) -> float:
    """Image level relative to the wanted signal (negative dB) for amplitude ratio g and phase error phi."""
    g, c = gain_ratio, np.cos(phase_rad)
    return float(-10 * np.log10((1 + 2 * g * c + g * g) / (1 - 2 * g * c + g * g)))


def summarise(
    stats: dict[str, np.ndarray], exclude: list[tuple[float, float]], seconds: float = 1.0
) -> dict[str, Any]:
    """Profile numbers from per-second stats; `exclude` ranges (s) are left out of the 'clean' figures."""
    keep = keep_mask(stats["t0"], exclude, pad_s=seconds) & keep_mask(stats["t0"] + seconds, exclude)
    mi, mq = hist_moments(stats["hist_i"]), hist_moments(stats["hist_q"])
    g = float(np.median(mq["rms"][keep]) / np.median(mi["rms"][keep]))
    rho = float(np.median(stats["cov_iq"][keep] / (mi["rms"][keep] * mq["rms"][keep])))
    phi = float(np.arcsin(np.clip(rho, -1, 1)))
    rms = np.sqrt((mi["rms"] ** 2 + mq["rms"] ** 2) / 2)
    sig = (mi["sigma_mad"] + mq["sigma_mad"]) / 2
    clip = (mi["clip"] + mq["clip"]) / 2
    tot = stats["hist_i"].sum(axis=0) + stats["hist_q"].sum(axis=0)
    return {
        "seconds_total": len(keep),
        "seconds_clean": int(keep.sum()),
        "dc_i_counts": float(np.median(mi["mean"][keep])),
        "dc_q_counts": float(np.median(mq["mean"][keep])),
        "rms_counts": float(np.median(rms[keep])),
        "noise_sigma_mad_counts": float(np.median(sig[keep])),
        "noise_sigma_mad_counts_p10_p90": [float(np.percentile(sig[keep], 10)), float(np.percentile(sig[keep], 90))],
        "rms_counts_all_p50_p99": [float(np.median(rms)), float(np.percentile(rms, 99))],
        "clip_fraction_median": float(np.median(clip)),
        "clip_fraction_mean": float(np.mean(clip)),
        "clip_fraction_max_1s": float(np.max(clip)),
        "value_fraction_0_1_254_255": [float(tot[v] / tot.sum()) for v in (0, 1, 254, 255)],
        "iq_gain_ratio": g,
        "iq_corr": rho,
        "iq_phase_deg": float(np.degrees(phi)),
        "image_rejection_from_stats_db": image_rejection_db(g, phi),
    }
