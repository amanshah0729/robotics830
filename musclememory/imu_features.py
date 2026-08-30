"""IMU signal processing: window resampling, handcrafted features, and derived
events (contact spikes, chapter boundaries).

Handcrafted features serve two jobs: the no-torch retrieval baseline, and the
honest yardstick the learned IMU encoder has to beat in eval.
"""

from __future__ import annotations

import numpy as np

from . import config

FEATURE_NAMES = [
    "rms_ax", "rms_ay", "rms_az",
    "mag_mean", "mag_std",
    "jerk_rms",
    "dom_freq_hz", "periodicity",
    "spec_centroid",
    "band_0.5_2", "band_2_5", "band_5_10",
    "spike_count",
    "gyro_rms",
    "zcr",
    "az_mean",
]


def resample_window(times: np.ndarray, data: np.ndarray,
                    t0: float, t1: float,
                    out_len: int = config.IMU_LEN,
                    out_ch: int = config.IMU_CHANNELS) -> np.ndarray:
    """Linear-resample a variable-rate window onto a fixed (out_len, out_ch) grid.
    Missing channels (e.g. no gyro) are zero-filled."""
    grid = np.linspace(t0, t1, out_len)
    out = np.zeros((out_len, out_ch), dtype=np.float32)
    if len(times) >= 2:
        c = min(data.shape[1], out_ch)
        for j in range(c):
            out[:, j] = np.interp(grid, times, data[:, j])
    return out


def _detrended_mag(data: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(data[:, :3], axis=1)
    return mag - mag.mean()


def featurize(times: np.ndarray, data: np.ndarray) -> np.ndarray:
    """~16-dim descriptor of one window. `data` is (m, C>=3), accel first."""
    m = len(data)
    if m < 8:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    dt = np.median(np.diff(times)) if m > 1 else 1.0 / config.ASSUMED_IMU_RATE
    rate = 1.0 / max(dt, 1e-6)

    acc = data[:, :3]
    mag = np.linalg.norm(acc, axis=1)
    dmag = mag - mag.mean()
    jerk = np.diff(mag) * rate

    # spectrum of detrended magnitude
    win = dmag * np.hanning(m)
    spec = np.abs(np.fft.rfft(win)) ** 2
    freqs = np.fft.rfftfreq(m, d=dt)
    band = (freqs >= 0.3) & (freqs <= 12.0)
    total = spec[band].sum() + 1e-9

    if band.any():
        bi = np.argmax(spec * band)
        dom_freq = float(freqs[bi])
        periodicity = float(spec[bi] / total)
        centroid = float((freqs[band] * spec[band]).sum() / total)
    else:
        dom_freq = periodicity = centroid = 0.0

    def band_energy(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return float(spec[sel].sum() / total)

    spikes = detect_spikes(times, acc)
    gyro_rms = float(np.sqrt((data[:, 3:6] ** 2).mean())) if data.shape[1] >= 6 else 0.0
    zcr = float((np.diff(np.signbit(dmag)) != 0).mean())

    f = np.array(
        [
            *np.sqrt((acc ** 2).mean(axis=0)),
            mag.mean(), mag.std(),
            np.sqrt((jerk ** 2).mean()) if len(jerk) else 0.0,
            dom_freq, periodicity, centroid,
            band_energy(0.5, 2), band_energy(2, 5), band_energy(5, 10),
            float(len(spikes)),
            gyro_rms, zcr, acc[:, 2].mean(),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(f)


def detect_spikes(times: np.ndarray, acc: np.ndarray,
                  z_thresh: float = 4.0, min_gap_s: float = 0.25) -> list[float]:
    """Contact/impact events: sharp jumps in |a| that exceed z_thresh sigmas."""
    if len(acc) < 8:
        return []
    mag = np.linalg.norm(acc[:, :3], axis=1)
    d = np.abs(np.diff(mag))
    sigma = d.std() + 1e-9
    idx = np.where(d > z_thresh * sigma)[0]
    out: list[float] = []
    for i in idx:
        t = float(times[i])
        if not out or t - out[-1] >= min_gap_s:
            out.append(t)
    return out


def chapter_boundaries(times: np.ndarray, acc: np.ndarray,
                       smooth_s: float = 1.0, min_seg_s: float = 3.0) -> list[float]:
    """Action boundaries at sustained low-motion valleys of smoothed |a| energy."""
    if len(acc) < 16:
        return []
    dt = np.median(np.diff(times))
    mag = np.abs(_detrended_mag(acc))
    k = max(1, int(smooth_s / max(dt, 1e-6)))
    kernel = np.ones(k) / k
    smooth = np.convolve(mag, kernel, mode="same")
    thresh = smooth.mean() - 0.5 * smooth.std()
    low = smooth < thresh

    bounds: list[float] = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if (j - i) * dt >= 0.5:  # sustained lull
                center = float(times[i + int(np.argmin(smooth[i:j]))])
                if not bounds or center - bounds[-1] >= min_seg_s:
                    bounds.append(center)
            i = j
        else:
            i += 1
    return bounds
