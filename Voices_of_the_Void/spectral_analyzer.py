"""
spectral_analyzer.py — Deep Spectral Analysis & Radio Astronomy Engine
Voices of the Void | Alpen Signal Observatorium — Dunkeltaler Forest
Array Frequency Domain Intelligence System v3.1.0

Handles: Waterfall generation, Fourier deep-dives, Doppler mechanics,
         SETI candidate scoring, radio astronomy parametrics, pulsar
         timing, spectral line identification, RFI excision,
         dynamic spectra, Stokes parameter estimation.
"""

from __future__ import annotations

import hashlib
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Iterator

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.fft as sp_fft
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, median_filter, label as ndimage_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS & SPECTRAL LINE CATALOG
# ─────────────────────────────────────────────────────────────────────────────

SPEED_OF_LIGHT_MS   = 2.99792458e8
HYDROGEN_21CM_MHZ   = 1420.405751768
OH_1612_MHZ         = 1612.231
OH_1665_MHZ         = 1665.4018
OH_1667_MHZ         = 1667.3590
OH_1720_MHZ         = 1720.5300
WATER_MASER_MHZ     = 22235.0800
METHANOL_MHZ        = 6668.5192
FORMALDEHYDE_MHZ    = 4829.6594
AMMONIA_MHZ         = 23694.4955
SIO_MASER_MHZ       = 43122.0800
HC3N_MHZ            = 9098.332
CS_MHZ              = 48990.957

SPECTRAL_LINE_CATALOG: Dict[str, float] = {
    "H-I 21cm":         HYDROGEN_21CM_MHZ,
    "OH 1612":          OH_1612_MHZ,
    "OH 1665 (main)":   OH_1665_MHZ,
    "OH 1667 (main)":   OH_1667_MHZ,
    "OH 1720":          OH_1720_MHZ,
    "H2O maser":        WATER_MASER_MHZ,
    "CH3OH maser":      METHANOL_MHZ,
    "H2CO":             FORMALDEHYDE_MHZ,
    "NH3":              AMMONIA_MHZ,
    "SiO maser":        SIO_MASER_MHZ,
    "HC3N":             HC3N_MHZ,
    "CS":               CS_MHZ,
}

WATER_HOLE = (HYDROGEN_21CM_MHZ, OH_1720_MHZ)

# Pulsar catalogue (B-name, period_s, DM)
KNOWN_PULSARS: List[Tuple[str, float, float]] = [
    ("B0531+21 (Crab)",   0.033102,   56.8),
    ("B0833-45 (Vela)",   0.089329,   67.9),
    ("B1919+21",          1.337301,   12.4),
    ("B0950+08",          0.253065,    2.9),
    ("B1257+12",          0.006219,   10.2),
    ("B1937+21",          0.001558,   71.0),
    ("B0329+54",          0.714519,   26.8),
    ("B1055-52",          0.197108,   30.1),
    ("B1509-58",          0.150658,  252.7),
    ("J0437-4715",        0.005757,    2.6),
    ("J0534+2200",        0.033092,   56.8),
    ("J1012+5307",        0.005256,    9.0),
    ("B1642-03",          0.387689,   35.7),
    ("B0823+26",          0.530661,   19.5),
    ("B2021+51",          0.529198,   22.6),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class RFIType(Enum):
    NARROWBAND_TONE     = "NARROWBAND_TONE"
    BROADBAND_IMPULSE   = "BROADBAND_IMPULSE"
    PERIODIC_HARMONIC   = "PERIODIC_HARMONIC"
    SWEPT_INTERFERENCE  = "SWEPT_INTERFERENCE"
    DIGITAL_ALIASING    = "DIGITAL_ALIASING"
    UNKNOWN             = "UNKNOWN"

@dataclass
class SpectralLine:
    frequency_mhz:     float
    amplitude_db:      float
    width_hz:          float
    line_name:         Optional[str] = None
    radial_vel_kms:    float = 0.0
    is_rfi:            bool = False
    rfi_type:          RFIType = RFIType.UNKNOWN
    significance:      float = 0.0         # sigma above noise
    notes:             str = ""

@dataclass
class WaterfallFrame:
    """Single STFT frame in a dynamic spectrum."""
    timestamp:     float
    freq_axis_hz:  np.ndarray
    power_db:      np.ndarray
    noise_floor:   float
    peak_freq_hz:  float
    peak_power_db: float
    rfi_mask:      np.ndarray   # Boolean — flagged channels

@dataclass
class DynamicSpectrum:
    """Full waterfall / dynamic spectrum object."""
    center_freq_mhz:  float
    sample_rate_hz:   float
    n_fft:            int
    hop_samples:      int
    window_name:      str
    time_axis_s:      np.ndarray
    freq_axis_hz:     np.ndarray
    power_db_matrix:  np.ndarray      # shape: (n_freq, n_time)
    rfi_mask:         np.ndarray      # shape: (n_freq, n_time)
    noise_floor_db:   float
    created_at:       float = field(default_factory=time.time)

    @property
    def n_time_bins(self) -> int:
        return len(self.time_axis_s)

    @property
    def n_freq_bins(self) -> int:
        return len(self.freq_axis_hz)

    @property
    def time_resolution_s(self) -> float:
        return float(self.hop_samples / self.sample_rate_hz)

    @property
    def freq_resolution_hz(self) -> float:
        return float(self.sample_rate_hz / self.n_fft)

    @property
    def cleaned_power(self) -> np.ndarray:
        """Return power matrix with RFI channels zeroed (set to noise floor)."""
        p = self.power_db_matrix.copy()
        p[self.rfi_mask] = self.noise_floor_db
        return p

@dataclass
class PulsarFit:
    candidate_period_s: float
    fitted_period_s:    float
    period_error_s:     float
    dm:                 float
    dm_error:           float
    matched_catalog:    Optional[str]
    profile_snr:        float
    n_folds:            int
    chi2_reduced:       float
    timing_residuals:   np.ndarray

@dataclass
class DopplerAnalysis:
    source_freq_mhz:   float
    observed_freq_mhz: float
    doppler_shift_hz:  float
    radial_vel_kms:    float
    vel_uncertainty_kms: float
    drift_rate_hz_s:   float
    acceleration_ms2:  Optional[float]
    lsr_correction_kms: float      # Local Standard of Rest
    barycentric_corr_hz: float

@dataclass
class SETICandidate:
    signal_id:        str
    timestamp:        float
    center_freq_mhz:  float
    bandwidth_hz:     float
    snr_db:           float
    drift_rate_hz_s:  float
    dispersion_measure: float
    wow_score:        float          # 0–10
    coherence_score:  float          # 0–1 — periodicity / structure
    extraterrestrial_probability: float  # 0–1, Bayesian
    rfi_rejection_passes: int
    notes:            str = ""
    classification:   str = "UNCONFIRMED"


# ─────────────────────────────────────────────────────────────────────────────
# CORE SPECTRAL ANALYZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SpectralAnalyzer:
    """
    Full-featured spectral analysis engine for radio astronomy.
    All methods are numerically rigorous; no cosmetic shortcuts.
    """

    WINDOW_REGISTRY = {
        "rectangular":      lambda n: np.ones(n),
        "hamming":          np.hamming,
        "hann":             np.hanning,
        "blackman":         np.blackman,
        "blackman_harris":  lambda n: sp_signal.windows.blackmanharris(n),
        "flattop":          lambda n: sp_signal.windows.flattop(n),
        "kaiser_5":         lambda n: np.kaiser(n, 5.0),
        "kaiser_8":         lambda n: np.kaiser(n, 8.0),
        "kaiser_14":        lambda n: np.kaiser(n, 14.0),
        "tukey_0.5":        lambda n: sp_signal.windows.tukey(n, 0.5),
        "nuttall":          lambda n: sp_signal.windows.nuttall(n),
        "dpss":             lambda n: sp_signal.windows.dpss(n, 4),
    }

    def __init__(self, sample_rate: float = 2e6, center_freq_mhz: float = 1420.0):
        self.sample_rate = sample_rate
        self.center_freq_mhz = center_freq_mhz
        self.nyquist = sample_rate / 2.0
        self._rfi_db_threshold = 15.0      # dB above noise floor to flag
        self._line_sigma_threshold = 5.0   # sigma above noise to call a line

    # ─── Window factory ───────────────────────────────────────────────────────

    def _window(self, name: str, n: int) -> np.ndarray:
        fn = self.WINDOW_REGISTRY.get(name, np.hanning)
        try:
            w = fn(n)
            if w.ndim > 1:     # dpss returns 2D
                w = w[0]
            return w.astype(np.float64)
        except Exception:
            return np.hanning(n)

    # ─── Noise estimation ────────────────────────────────────────────────────

    def estimate_noise_floor(self, power_db: np.ndarray, method: str = "mad") -> float:
        """
        Estimate noise floor using robust statistics.
        Methods: 'mad' (median absolute deviation), 'percentile', 'iterative_sigma'.
        """
        p = power_db[np.isfinite(power_db)]
        if method == "mad":
            median = np.median(p)
            mad = np.median(np.abs(p - median))
            sigma = 1.4826 * mad        # Gaussian-equivalent sigma
            return float(median - 2 * sigma)
        elif method == "percentile":
            return float(np.percentile(p, 10))
        elif method == "iterative_sigma":
            mu, sig = np.mean(p), np.std(p)
            for _ in range(5):
                mask = np.abs(p - mu) < 3 * sig
                if mask.sum() < 10:
                    break
                mu = np.mean(p[mask])
                sig = np.std(p[mask])
            return float(mu - 2 * sig)
        return float(np.percentile(p, 10))

    def noise_power_density(self, power_db: np.ndarray) -> Tuple[float, float]:
        """
        Returns (noise_floor_db, noise_std_db).
        Uses iterative sigma-clipping on the power spectrum.
        """
        p = power_db[np.isfinite(power_db)]
        mu, sig = np.mean(p), np.std(p)
        for _ in range(8):
            mask = p < mu + 2.5 * sig
            if mask.sum() < 5:
                break
            mu = np.mean(p[mask])
            sig = np.std(p[mask])
        return float(mu), float(sig)

    # ─── Core FFT with proper calibration ────────────────────────────────────

    def calibrated_fft(self, signal: np.ndarray, n_fft: Optional[int] = None,
                       window: str = "blackman_harris",
                       return_one_sided: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrated FFT returning absolute power in dBFS (dB relative to full scale).
        Corrects for window coherent gain.
        Returns (freq_hz_axis, power_dbfs).
        """
        n = len(signal)
        if n_fft is None:
            n_fft = n
        w = self._window(window, n)
        # Coherent gain (for amplitude calibration)
        cg = np.mean(w)
        # Incoherent power gain (for power calibration)
        ipg = np.mean(w ** 2)
        windowed = signal[:n] * w
        S = sp_fft.fft(windowed, n=n_fft)
        freqs = sp_fft.fftfreq(n_fft, d=1.0 / self.sample_rate)
        # Power spectrum — amplitude-squared, normalized
        power = (np.abs(S) ** 2) / (n ** 2 * ipg)
        power_db = 10 * np.log10(power + 1e-30)
        if return_one_sided:
            half = n_fft // 2
            freqs = freqs[:half]
            power_db = power_db[:half]
            power_db[1:-1] += 3.0   # Correct for one-sided doubling
        return freqs, power_db

    def averaged_periodogram(self, signal: np.ndarray, n_fft: int = 1024,
                             n_overlap: int = 512, window: str = "hann",
                             n_avg: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Welch's method — averaged periodogram with optional max-hold.
        Returns (freq_hz, avg_power_db).
        """
        n = len(signal)
        hop = n_fft - n_overlap
        frames = [(i * hop, i * hop + n_fft)
                  for i in range((n - n_fft) // hop + 1)
                  if i * hop + n_fft <= n]
        if not frames:
            return self.calibrated_fft(signal, n_fft, window)
        if n_avg is not None:
            frames = frames[-n_avg:]
        w = self._window(window, n_fft)
        ipg = np.mean(w ** 2)
        spectra = []
        for s, e in frames:
            seg = signal[s:e] * w
            S = sp_fft.fft(seg, n=n_fft)
            power = np.abs(S[:n_fft // 2]) ** 2 / (n_fft ** 2 * ipg)
            spectra.append(power)
        avg_power = np.mean(spectra, axis=0)
        freqs = sp_fft.fftfreq(n_fft, d=1.0 / self.sample_rate)[:n_fft // 2]
        avg_power_db = 10 * np.log10(avg_power + 1e-30)
        avg_power_db[1:] += 3.0
        return freqs, avg_power_db

    # ─── Dynamic Spectrum / Waterfall ─────────────────────────────────────────

    def build_dynamic_spectrum(self, signal: np.ndarray,
                               n_fft: int = 512,
                               hop: int = 128,
                               window: str = "blackman_harris",
                               excise_rfi: bool = True
                               ) -> DynamicSpectrum:
        """
        Build full dynamic spectrum (waterfall) with RFI masking.
        """
        n = len(signal)
        n_frames = max(1, (n - n_fft) // hop + 1)
        w = self._window(window, n_fft)
        ipg = np.mean(w ** 2)
        half = n_fft // 2
        freqs = sp_fft.fftfreq(n_fft, d=1.0 / self.sample_rate)[:half]

        power_matrix = np.full((half, n_frames), -200.0)
        for i in range(n_frames):
            s, e = i * hop, i * hop + n_fft
            if e > n:
                seg = np.concatenate([signal[s:], np.zeros(e - n, dtype=complex)])
            else:
                seg = signal[s:e]
            seg_w = seg * w
            S = sp_fft.fft(seg_w, n=n_fft)
            power = np.abs(S[:half]) ** 2 / (n_fft ** 2 * ipg)
            power_matrix[:, i] = 10 * np.log10(power + 1e-30)

        time_axis = np.arange(n_frames) * hop / self.sample_rate
        noise_floor, _ = self.noise_power_density(power_matrix.ravel())

        # RFI masking
        rfi_mask = np.zeros_like(power_matrix, dtype=bool)
        if excise_rfi:
            rfi_mask = self._rfi_excision_2d(power_matrix, noise_floor)

        return DynamicSpectrum(
            center_freq_mhz=self.center_freq_mhz,
            sample_rate_hz=self.sample_rate,
            n_fft=n_fft,
            hop_samples=hop,
            window_name=window,
            time_axis_s=time_axis,
            freq_axis_hz=freqs,
            power_db_matrix=power_matrix,
            rfi_mask=rfi_mask,
            noise_floor_db=float(noise_floor),
        )

    def _rfi_excision_2d(self, power_matrix: np.ndarray, noise_floor: float) -> np.ndarray:
        """
        Multi-pass 2D RFI excision:
        1. Spectral kurtosis flagging (impulsive RFI)
        2. Narrowband persistent tone flagging
        3. Broadband impulse flagging
        """
        n_freq, n_time = power_matrix.shape
        mask = np.zeros((n_freq, n_time), dtype=bool)

        # Pass 1: Per-channel spectral kurtosis
        # SK = (N*S4/S2^2 - 1) * (N+1)/(N-1) ; expect SK=1 for Gaussian noise
        lin = 10 ** (power_matrix / 10)
        s1 = np.mean(lin, axis=1)
        s2 = np.mean(lin ** 2, axis=1)
        n = n_time
        with np.errstate(invalid="ignore", divide="ignore"):
            sk = np.where(s1 > 0, (n * s2 / (s1 ** 2) - 1) * (n + 1) / max(n - 1, 1), 1.0)
        # Channels with SK far from 1 are RFI
        sk_mean, sk_std = np.median(sk), np.std(sk[np.isfinite(sk)])
        rfi_channels = np.abs(sk - 1.0) > (4.0 * sk_std + 0.5)
        mask[rfi_channels, :] = True

        # Pass 2: Persistent narrowband — channels that are bright in most time bins
        threshold_db = noise_floor + self._rfi_db_threshold
        persistent = np.mean(power_matrix > threshold_db, axis=1) > 0.6
        mask[persistent, :] = True

        # Pass 3: Broadband impulse — time bins bright across most channels
        broadband_impulse = np.mean(power_matrix > threshold_db, axis=0) > 0.5
        mask[:, broadband_impulse] = True

        return mask

    # ─── Spectral Line Detection ──────────────────────────────────────────────

    def detect_spectral_lines(self, freqs: np.ndarray, power_db: np.ndarray,
                              n_sigma: float = 5.0,
                              min_separation_hz: float = 500.0,
                              max_lines: int = 50
                              ) -> List[SpectralLine]:
        """
        Detect spectral emission/absorption lines using sigma thresholding
        with iterative source subtraction and Gaussian fitting.
        """
        noise_floor, noise_sigma = self.noise_power_density(power_db)
        threshold_db = noise_floor + n_sigma * noise_sigma
        power_lin = 10 ** (power_db / 10)
        noise_lin = 10 ** (noise_floor / 10)

        # Find peaks above threshold
        min_sep_bins = max(1, int(min_separation_hz / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 1
        peaks, properties = sp_signal.find_peaks(
            power_db,
            height=threshold_db,
            distance=min_sep_bins,
            prominence=noise_sigma * 2.0,
        )

        lines: List[SpectralLine] = []
        for pk in peaks[:max_lines]:
            freq_pk = float(freqs[pk])
            amp_pk = float(power_db[pk])
            significance = (amp_pk - noise_floor) / (noise_sigma + 1e-10)

            # Gaussian fit to estimate width
            fit_half = min(30, pk, len(freqs) - 1 - pk)
            fit_slice = slice(pk - fit_half, pk + fit_half + 1)
            fit_freqs = freqs[fit_slice]
            fit_power = power_lin[fit_slice]
            width_hz = self._fit_gaussian_width(fit_freqs, fit_power)

            # Catalog match
            line_name = self._match_spectral_line(freq_pk, tolerance_hz=5e4)

            # Radial velocity (relative to H-I rest frequency)
            doppler_freq = self.center_freq_mhz * 1e6 + freq_pk
            v_rad = self._freq_to_velocity(doppler_freq, HYDROGEN_21CM_MHZ * 1e6)

            # RFI heuristic
            is_rfi = self._is_likely_rfi(freq_pk, amp_pk, noise_floor, noise_sigma)

            lines.append(SpectralLine(
                frequency_mhz=doppler_freq / 1e6,
                amplitude_db=amp_pk,
                width_hz=width_hz,
                line_name=line_name,
                radial_vel_kms=v_rad,
                is_rfi=is_rfi,
                rfi_type=RFIType.NARROWBAND_TONE if is_rfi else RFIType.UNKNOWN,
                significance=float(significance),
            ))

        lines.sort(key=lambda l: -l.significance)
        return lines

    def _fit_gaussian_width(self, freqs: np.ndarray, power_lin: np.ndarray) -> float:
        """Fit Gaussian to estimate spectral line width (FWHM in Hz)."""
        if len(freqs) < 5:
            return float(freqs[-1] - freqs[0]) if len(freqs) > 1 else 0.0
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        try:
            peak_idx = np.argmax(power_lin)
            p0 = [power_lin[peak_idx], freqs[peak_idx],
                  (freqs[-1] - freqs[0]) / 4.0]
            popt, _ = curve_fit(gaussian, freqs, power_lin, p0=p0,
                                maxfev=500, bounds=(0, np.inf))
            return float(2.355 * abs(popt[2]))  # FWHM
        except Exception:
            return float(freqs[-1] - freqs[0]) if len(freqs) > 1 else 0.0

    def _match_spectral_line(self, freq_offset_hz: float, tolerance_hz: float = 5e4) -> Optional[str]:
        """Check if a detected frequency offset matches a known spectral line."""
        obs_freq_mhz = self.center_freq_mhz + freq_offset_hz / 1e6
        for name, rest_mhz in SPECTRAL_LINE_CATALOG.items():
            # Allow for ±500 km/s Doppler shift
            max_shift_mhz = rest_mhz * 500 / (SPEED_OF_LIGHT_MS / 1e3)
            if abs(obs_freq_mhz - rest_mhz) < max_shift_mhz + tolerance_hz / 1e6:
                return name
        return None

    @staticmethod
    def _freq_to_velocity(observed_hz: float, rest_hz: float) -> float:
        """Compute radial velocity (km/s) from Doppler shift."""
        return float((rest_hz - observed_hz) / rest_hz * SPEED_OF_LIGHT_MS / 1e3)

    def _is_likely_rfi(self, freq_offset_hz: float, amp_db: float,
                       noise_floor: float, noise_sigma: float) -> bool:
        """
        Heuristic RFI rejection.
        Strong signals at baseband edges, or exact integer Hz offsets, are flagged.
        """
        nyq = self.sample_rate / 2
        # Edge channels
        if abs(freq_offset_hz) > nyq * 0.95:
            return True
        # Suspiciously strong relative to noise
        if (amp_db - noise_floor) > 60 * noise_sigma:
            return True
        # Exact MHz multiples (power-supply harmonics, oscillator spurs)
        abs_freq_hz = abs(self.center_freq_mhz * 1e6 + freq_offset_hz)
        if abs(abs_freq_hz % 1e6) < 1000 or abs(abs_freq_hz % 5e5) < 500:
            return True
        return False

    # ─── Doppler Analysis ─────────────────────────────────────────────────────

    def compute_doppler_analysis(self, signal: np.ndarray,
                                  rest_freq_mhz: float,
                                  time_segments: int = 32,
                                  observer_vel_kms: float = 0.0
                                  ) -> DopplerAnalysis:
        """
        Full Doppler analysis:
        - Observed frequency vs rest frequency
        - Radial velocity with LSR correction
        - Drift rate via linear regression
        - Acceleration estimate from quadratic fit
        - Barycentric correction (simplified)
        """
        n = len(signal)
        seg_len = n // time_segments
        peak_freqs_hz: List[float] = []
        for i in range(time_segments):
            seg = signal[i * seg_len:(i + 1) * seg_len]
            f, p = self.calibrated_fft(seg, window="hann")
            if len(p) > 0:
                peak_freqs_hz.append(float(f[np.argmax(p)]))

        if len(peak_freqs_hz) < 4:
            observed_offset = peak_freqs_hz[0] if peak_freqs_hz else 0.0
            drift = 0.0
            accel = None
        else:
            seg_dur = seg_len / self.sample_rate
            times = np.arange(len(peak_freqs_hz)) * seg_dur
            # Linear fit for drift rate
            lin_coeffs = np.polyfit(times, peak_freqs_hz, 1)
            drift = float(lin_coeffs[0])      # Hz/s
            # Quadratic fit for acceleration
            quad_coeffs = np.polyfit(times, peak_freqs_hz, 2)
            accel_hz_s2 = float(quad_coeffs[0])
            # Convert frequency acceleration to physical acceleration
            c = SPEED_OF_LIGHT_MS
            rest_hz = rest_freq_mhz * 1e6
            accel = float(accel_hz_s2 * c / rest_hz)  # m/s²
            observed_offset = float(np.mean(peak_freqs_hz))

        obs_freq_hz = self.center_freq_mhz * 1e6 + observed_offset
        rest_hz = rest_freq_mhz * 1e6
        doppler_shift = float(obs_freq_hz - rest_hz)

        # Relativistic Doppler radial velocity
        beta = doppler_shift / rest_hz
        if abs(beta) < 0.99:
            v_rad = float(((1 + beta) ** 2 - 1) / ((1 + beta) ** 2 + 1) * SPEED_OF_LIGHT_MS / 1e3)
        else:
            v_rad = float(beta * SPEED_OF_LIGHT_MS / 1e3)

        # LSR correction (simplified: use observer_vel_kms as proxy)
        lsr_corr = observer_vel_kms
        v_lsr = v_rad - lsr_corr

        # Barycentric correction (simplified Earths orbital component ~30 km/s)
        bary_corr_hz = float(30e3 / SPEED_OF_LIGHT_MS * rest_hz)

        # Velocity uncertainty from fit residuals
        pf_arr = np.array(peak_freqs_hz)
        residuals = pf_arr - (np.polyval(lin_coeffs if len(peak_freqs_hz) >= 2 else [0, observed_offset],
                                          np.arange(len(pf_arr)) * (seg_len / self.sample_rate)))
        vel_unc = float(np.std(residuals) / rest_hz * SPEED_OF_LIGHT_MS / 1e3) if len(residuals) > 1 else 0.0

        return DopplerAnalysis(
            source_freq_mhz=rest_freq_mhz,
            observed_freq_mhz=float(obs_freq_hz / 1e6),
            doppler_shift_hz=doppler_shift,
            radial_vel_kms=v_lsr,
            vel_uncertainty_kms=vel_unc,
            drift_rate_hz_s=drift if len(peak_freqs_hz) >= 2 else 0.0,
            acceleration_ms2=accel,
            lsr_correction_kms=lsr_corr,
            barycentric_corr_hz=bary_corr_hz,
        )

    # ─── Pulsar Analysis ──────────────────────────────────────────────────────

    def fold_pulsar(self, signal: np.ndarray, period_s: float,
                    n_bins: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fold signal at given period to build integrated pulse profile.
        Returns (phase_bins, folded_profile_power).
        """
        env = np.abs(signal) ** 2
        n = len(env)
        n_samples_per_period = int(period_s * self.sample_rate)
        if n_samples_per_period < 2:
            return np.linspace(0, 1, n_bins), np.zeros(n_bins)
        n_folds = n // n_samples_per_period
        profile = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        bin_edges = np.linspace(0, n_samples_per_period, n_bins + 1, dtype=int)
        for fold_idx in range(n_folds):
            start = fold_idx * n_samples_per_period
            period_data = env[start:start + n_samples_per_period]
            for b in range(n_bins):
                b_start, b_end = bin_edges[b], bin_edges[b + 1]
                if b_end <= len(period_data):
                    profile[b] += np.mean(period_data[b_start:b_end])
                    counts[b] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.where(counts > 0, profile / counts, 0.0)
        # Baseline subtract
        profile -= np.percentile(profile, 20)
        phase = np.linspace(0, 1, n_bins, endpoint=False)
        return phase, profile

    def period_search_ffa(self, signal: np.ndarray,
                          p_min_s: float = 0.001,
                          p_max_s: float = 10.0,
                          n_candidates: int = 10
                          ) -> List[Tuple[float, float]]:
        """
        Fast Folding Algorithm (FFA) period search.
        Returns list of (period_s, snr) sorted by SNR descending.
        """
        env = np.abs(signal) ** 2
        n = len(env)
        # Trial periods on a log-spaced grid
        n_trials = min(2000, max(100, int(p_max_s / p_min_s * 20)))
        trial_periods = np.logspace(np.log10(p_min_s), np.log10(p_max_s), n_trials)
        results: List[Tuple[float, float]] = []
        noise_baseline = float(np.std(env))

        for p in trial_periods:
            n_per_period = int(p * self.sample_rate)
            if n_per_period < 4 or n_per_period > n:
                continue
            n_folds = n // n_per_period
            if n_folds < 2:
                continue
            folded = np.zeros(n_per_period)
            for k in range(n_folds):
                folded += env[k * n_per_period:(k + 1) * n_per_period]
            folded /= n_folds
            peak = float(np.max(folded) - np.mean(folded))
            snr = peak / (noise_baseline / math.sqrt(n_folds) + 1e-20)
            results.append((float(p), snr))

        results.sort(key=lambda x: -x[1])
        return results[:n_candidates]

    def fit_pulsar_candidate(self, signal: np.ndarray, candidate_period_s: float,
                             dm: float = 0.0) -> PulsarFit:
        """
        Refine pulsar candidate period via chi² minimization.
        Returns full PulsarFit with timing statistics.
        """
        n_bins = 64
        # Fine grid search around candidate
        n_fine = 51
        p_lo = candidate_period_s * 0.98
        p_hi = candidate_period_s * 1.02
        fine_periods = np.linspace(p_lo, p_hi, n_fine)
        best_snr, best_period = 0.0, candidate_period_s
        for p in fine_periods:
            _, prof = self.fold_pulsar(signal, p, n_bins)
            if prof.max() > 0:
                snr = float(prof.max() / (prof.std() + 1e-20))
                if snr > best_snr:
                    best_snr = snr
                    best_period = p

        _, best_profile = self.fold_pulsar(signal, best_period, n_bins)
        # Timing residuals: compare folded profile to Gaussian template
        phase = np.linspace(0, 1, n_bins)
        if best_profile.max() > 0:
            peak_phase = phase[np.argmax(best_profile)]
            residuals = (best_profile - best_profile.mean()) / (best_profile.std() + 1e-10)
        else:
            residuals = np.zeros(n_bins)
            peak_phase = 0.0

        # Chi² of fit to Gaussian template
        template = np.exp(-0.5 * ((phase - peak_phase) / 0.05) ** 2)
        template = (template - template.mean()) / (template.std() + 1e-10)
        dof = max(1, n_bins - 3)
        chi2_val = float(np.sum((residuals - template) ** 2) / dof)

        # Match to known pulsar catalog
        matched = None
        for name, p_cat, dm_cat in KNOWN_PULSARS:
            if (abs(best_period - p_cat) / p_cat < 0.01 and
                    (dm == 0 or abs(dm - dm_cat) / max(dm_cat, 1) < 0.3)):
                matched = name
                break

        n_folds = int(len(signal) / (best_period * self.sample_rate))
        period_error = best_period * 0.01 / max(n_folds, 1)

        return PulsarFit(
            candidate_period_s=candidate_period_s,
            fitted_period_s=best_period,
            period_error_s=period_error,
            dm=dm,
            dm_error=dm * 0.05,
            matched_catalog=matched,
            profile_snr=best_snr,
            n_folds=n_folds,
            chi2_reduced=chi2_val,
            timing_residuals=residuals,
        )

    # ─── De-dispersion ────────────────────────────────────────────────────────

    def incoherent_dedispersion(self, dynamic_spec: DynamicSpectrum,
                                dm_trials: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incoherent de-dispersion across DM trial values.
        Returns (dm_trials, dedispersed_snr_array).
        For each DM, shift frequency channels by dispersive delay and
        sum to form a dedispersed time series, then compute SNR.
        """
        power = dynamic_spec.cleaned_power     # (n_freq, n_time)
        freqs_mhz = self.center_freq_mhz + dynamic_spec.freq_axis_hz / 1e6
        n_freq, n_time = power.shape
        dt = dynamic_spec.time_resolution_s
        snr_vs_dm = np.zeros(len(dm_trials))

        for di, dm in enumerate(dm_trials):
            dedispersed = np.zeros(n_time)
            weights = 0
            for fi in range(n_freq):
                f_mhz = freqs_mhz[fi]
                if f_mhz <= 0:
                    continue
                # Dispersion delay relative to highest frequency channel
                f_hi_mhz = np.max(freqs_mhz[freqs_mhz > 0])
                delay_s = 4.148808e3 * dm * (1 / f_mhz ** 2 - 1 / f_hi_mhz ** 2)
                delay_bins = int(round(delay_s / dt))
                if 0 <= delay_bins < n_time:
                    shifted = np.roll(power[fi, :], -delay_bins)
                    dedispersed += shifted
                    weights += 1
            if weights > 0:
                dedispersed /= weights
            noise = float(np.std(dedispersed))
            snr_vs_dm[di] = float(np.max(dedispersed) / (noise + 1e-20))

        return dm_trials, snr_vs_dm

    # ─── SETI Candidate Scoring ───────────────────────────────────────────────

    def score_seti_candidate(self, signal: np.ndarray,
                              record_meta: Dict[str, Any]) -> SETICandidate:
        """
        Multi-criteria SETI candidate scoring.
        Implements a Bayesian framework combining multiple discriminants.
        """
        freqs, pdb = self.calibrated_fft(signal, window="blackman_harris")
        noise_floor, noise_sigma = self.noise_power_density(pdb)
        peak_idx = np.argmax(pdb)
        peak_db = float(pdb[peak_idx])
        snr_db = (peak_db - noise_floor) / (noise_sigma + 1e-10)

        # 1. Bandwidth criterion (narrow = more SETI-like)
        bw_mask = pdb > noise_floor + 3 * noise_sigma
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        bw_hz = float(np.sum(bw_mask)) * df
        bw_score = max(0.0, min(1.0, 1.0 - math.log10(max(bw_hz, 1)) / 7.0))

        # 2. Water hole proximity
        obs_freq_mhz = record_meta.get("center_freq_mhz", self.center_freq_mhz)
        if WATER_HOLE[0] <= obs_freq_mhz <= WATER_HOLE[1]:
            wh_score = 1.0
        else:
            dist = min(abs(obs_freq_mhz - WATER_HOLE[0]),
                       abs(obs_freq_mhz - WATER_HOLE[1]))
            wh_score = max(0.0, 1.0 - dist / 300.0)

        # 3. Drift rate plausibility
        doppler = self.compute_doppler_analysis(signal, obs_freq_mhz)
        drift = abs(doppler.drift_rate_hz_s)
        # Interstellar sources expected ~0.01–0.3 Hz/s after Earth rotation removal
        if 0.001 < drift < 0.5:
            drift_score = 1.0
        elif drift < 2.0:
            drift_score = 0.6
        elif drift < 5.0:
            drift_score = 0.2
        else:
            drift_score = 0.0       # Likely RFI or local transmitter

        # 4. Spectral purity (narrowband = narrow autocorrelation peak)
        pwr_lin = 10 ** (pdb / 10)
        total_power = np.sum(pwr_lin)
        peak_power = float(pwr_lin[peak_idx]) if len(pwr_lin) > 0 else 0
        purity = float(peak_power / (total_power + 1e-30))

        # 5. RFI rejection passes (multi-look)
        lines = self.detect_spectral_lines(freqs, pdb)
        n_rfi = sum(1 for l in lines if l.is_rfi)
        total_lines = max(len(lines), 1)
        rfi_rejection_passes = int((1 - n_rfi / total_lines) * 4)

        # 6. Coherence / periodicity
        env = np.abs(signal) ** 2
        env_fft = np.abs(sp_fft.rfft(env - np.mean(env))) ** 2
        if len(env_fft) > 1:
            coherence = float(np.max(env_fft[1:]) / (np.mean(env_fft[1:]) + 1e-20))
            coherence = min(1.0, coherence / 10.0)
        else:
            coherence = 0.0

        # 7. DM assessment
        dm = record_meta.get("dispersion_measure", 0.0)
        if dm > 5:     # Non-local source
            dm_score = min(1.0, dm / 200.0)
        else:
            dm_score = 0.1  # Could be local RFI (no dispersion)

        # Bayesian combination
        # Prior probability of ET signal is ~1e-6; update with likelihoods
        log_prior = math.log(1e-6)
        log_likelihood = (
            2.0 * math.log(max(snr_db / 10.0, 1e-6)) +
            3.0 * math.log(max(bw_score, 1e-6)) +
            2.0 * math.log(max(wh_score + 0.01, 1e-6)) +
            1.5 * math.log(max(drift_score + 0.01, 1e-6)) +
            1.0 * math.log(max(purity + 0.01, 1e-6)) +
            1.0 * math.log(max(dm_score + 0.01, 1e-6)) +
            1.0 * math.log(max(coherence + 0.01, 1e-6))
        )
        log_posterior = log_prior + log_likelihood
        # Softmax-style probability bounded to [0,1]
        et_probability = float(1.0 / (1.0 + math.exp(-log_posterior / 5.0)))

        # WoW score (0–10)
        wow = min(10.0, (
            snr_db / 30.0 * 2.5 +
            bw_score * 1.5 +
            wh_score * 1.5 +
            drift_score * 1.0 +
            purity * 0.5 +
            coherence * 1.0 +
            dm_score * 0.5 +
            rfi_rejection_passes * 0.5
        ))

        candidate_notes = []
        if obs_freq_mhz == HYDROGEN_21CM_MHZ:
            candidate_notes.append("EXACT H-I REST FREQUENCY — MAXIMUM INTEREST")
        if drift_score == 1.0:
            candidate_notes.append("DRIFT RATE CONSISTENT WITH STELLAR COMPANION")
        if dm > 100:
            candidate_notes.append(f"HIGH DM={dm:.1f} — EXTRAGALACTIC ORIGIN POSSIBLE")
        if et_probability > 0.01:
            candidate_notes.append("BAYESIAN POSTERIOR ELEVATED — REQUIRES FOLLOW-UP")

        classification = (
            "CONFIRMED_ASTROPHYSICAL" if (dm > 20 and bw_hz > 1e5) else
            "SETI_HIGH_PRIORITY"      if (wow > 7 and et_probability > 0.01) else
            "SETI_CANDIDATE"          if wow > 4 else
            "LIKELY_RFI"              if n_rfi > 0 else
            "UNCONFIRMED"
        )

        return SETICandidate(
            signal_id=record_meta.get("uid", "UNKNOWN"),
            timestamp=time.time(),
            center_freq_mhz=float(obs_freq_mhz),
            bandwidth_hz=float(bw_hz),
            snr_db=float(snr_db),
            drift_rate_hz_s=float(doppler.drift_rate_hz_s),
            dispersion_measure=float(dm),
            wow_score=float(round(wow, 3)),
            coherence_score=float(coherence),
            extraterrestrial_probability=float(et_probability),
            rfi_rejection_passes=rfi_rejection_passes,
            notes="; ".join(candidate_notes),
            classification=classification,
        )

    # ─── Stokes Parameters (polarimetry) ─────────────────────────────────────

    def compute_stokes(self, signal_x: np.ndarray,
                       signal_y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Stokes parameters I, Q, U, V from dual-polarization inputs.
        I = |X|² + |Y|²        (total intensity)
        Q = |X|² − |Y|²        (linear polarization, X−Y basis)
        U = 2·Re(X·Y*)          (linear polarization, ±45° basis)
        V = 2·Im(X·Y*)          (circular polarization)
        All returned as time-series arrays.
        """
        I = np.abs(signal_x) ** 2 + np.abs(signal_y) ** 2
        Q = np.abs(signal_x) ** 2 - np.abs(signal_y) ** 2
        cross = signal_x * np.conj(signal_y)
        U = 2 * np.real(cross)
        V = 2 * np.imag(cross)
        return I, Q, U, V

    def polarization_fraction(self, I: np.ndarray, Q: np.ndarray,
                               U: np.ndarray, V: np.ndarray
                               ) -> Tuple[float, float, float]:
        """
        Returns (linear_fraction, circular_fraction, total_polarization_fraction).
        Averaged over the time series.
        """
        I_mean = float(np.mean(I))
        if I_mean == 0:
            return 0.0, 0.0, 0.0
        lin = float(np.mean(np.sqrt(Q ** 2 + U ** 2))) / I_mean
        circ = float(np.mean(np.abs(V))) / I_mean
        total = float(np.mean(np.sqrt(Q ** 2 + U ** 2 + V ** 2))) / I_mean
        return lin, circ, total

    # ─── Spectral Kurtosis ────────────────────────────────────────────────────

    def spectral_kurtosis(self, signal: np.ndarray, n_fft: int = 256,
                          n_acc: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-channel spectral kurtosis for RFI and anomaly detection.
        SK ≈ 1 for noise; SK >> 1 for impulsive RFI; SK < 1 for CW.
        Returns (freq_hz, SK_array).
        """
        n = len(signal)
        hop = n_fft
        frames = []
        for i in range(min(n_acc * 2, n // n_fft)):
            seg = signal[i * hop:i * hop + n_fft]
            w = self._window("hann", n_fft)
            S = sp_fft.fft(seg * w, n=n_fft)
            power = np.abs(S[:n_fft // 2]) ** 2
            frames.append(power)
        if len(frames) < 4:
            freqs = sp_fft.fftfreq(n_fft, 1.0 / self.sample_rate)[:n_fft // 2]
            return freqs, np.ones(n_fft // 2)

        frames_arr = np.array(frames)  # (n_frames, n_freq)
        n_f = frames_arr.shape[0]
        S1 = np.mean(frames_arr, axis=0)
        S2 = np.mean(frames_arr ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            SK = np.where(S1 > 0,
                          (n_f * S2 / (S1 ** 2) - 1) * (n_f + 1) / max(n_f - 1, 1),
                          1.0)
        freqs = sp_fft.fftfreq(n_fft, 1.0 / self.sample_rate)[:n_fft // 2]
        return freqs, SK

    # ─── Signal Band Power ────────────────────────────────────────────────────

    def band_power(self, signal: np.ndarray,
                   bands: Dict[str, Tuple[float, float]]
                   ) -> Dict[str, float]:
        """
        Compute power in named frequency bands.
        bands: {'band_name': (low_hz, high_hz), ...}
        Returns {'band_name': power_dB, ...}
        """
        freqs, pdb = self.calibrated_fft(signal, window="blackman_harris")
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        results: Dict[str, float] = {}
        for band_name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.any():
                pwr_lin = 10 ** (pdb[mask] / 10)
                pwr_total = float(np.sum(pwr_lin) * df)
                results[band_name] = float(10 * np.log10(pwr_total + 1e-30))
            else:
                results[band_name] = -200.0
        return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SpectralVisualizer:
    """All plot methods return matplotlib Figure objects for st.pyplot()."""

    PALETTE = {
        "bg":       "#030d07",
        "fg":       "#00ff88",
        "fg_dim":   "#004422",
        "accent":   "#00ffcc",
        "warn":     "#ff8800",
        "danger":   "#ff2222",
        "grid":     "#001a0e",
        "axis":     "#335544",
    }

    def __init__(self):
        plt.rcParams.update({
            "font.family":      "monospace",
            "axes.facecolor":   self.PALETTE["bg"],
            "figure.facecolor": self.PALETTE["bg"],
            "axes.edgecolor":   self.PALETTE["fg_dim"],
            "axes.labelcolor":  self.PALETTE["fg"],
            "xtick.color":      self.PALETTE["axis"],
            "ytick.color":      self.PALETTE["axis"],
            "text.color":       self.PALETTE["fg"],
            "grid.color":       self.PALETTE["grid"],
            "grid.linestyle":   ":",
            "grid.alpha":       0.5,
            "axes.titlesize":   8,
            "axes.labelsize":   7,
            "xtick.labelsize":  6,
            "ytick.labelsize":  6,
        })

    def _style_ax(self, ax: plt.Axes, title: str = "") -> None:
        ax.set_facecolor(self.PALETTE["bg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(self.PALETTE["fg_dim"])
        if title:
            ax.set_title(title, color=self.PALETTE["fg"], fontsize=8,
                         loc="left", pad=4)
        ax.grid(True, alpha=0.3, color=self.PALETTE["grid"])

    # ─── Power spectrum ───────────────────────────────────────────────────────

    def plot_power_spectrum(self, freqs: np.ndarray, power_db: np.ndarray,
                            lines: Optional[List[SpectralLine]] = None,
                            title: str = "POWER SPECTRUM",
                            noise_floor: Optional[float] = None,
                            figsize: Tuple[float, float] = (10, 3.5)
                            ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(freqs / 1e3, power_db, color=self.PALETTE["fg"],
                lw=0.7, alpha=0.95, rasterized=True)
        ax.fill_between(freqs / 1e3, power_db,
                        np.full_like(power_db, power_db.min()),
                        alpha=0.1, color=self.PALETTE["fg"])
        if noise_floor is not None:
            ax.axhline(noise_floor, color=self.PALETTE["warn"],
                       ls="--", lw=0.8, alpha=0.7, label="Noise floor")
            ax.axhline(noise_floor + 5, color=self.PALETTE["accent"],
                       ls=":", lw=0.6, alpha=0.5, label="5σ threshold")

        if lines:
            for line in lines[:10]:
                f_off = (line.frequency_mhz * 1e6 - freqs[0] * 0 +
                         (line.frequency_mhz * 1e6 - np.mean(freqs) * 0)) / 1e3
                col = self.PALETTE["danger"] if line.is_rfi else self.PALETTE["accent"]
                ax.axvline(f_off / 1e3 if line.frequency_mhz < 1e3 else f_off,
                           color=col, lw=0.6, alpha=0.6)

        self._style_ax(ax, title)
        ax.set_xlabel("Freq offset (kHz)")
        ax.set_ylabel("Power (dBFS)")
        ax.legend(fontsize=6, loc="upper right",
                  facecolor=self.PALETTE["bg"],
                  edgecolor=self.PALETTE["fg_dim"],
                  labelcolor=self.PALETTE["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    # ─── Waterfall ────────────────────────────────────────────────────────────

    def plot_waterfall(self, ds: DynamicSpectrum,
                       use_cleaned: bool = True,
                       figsize: Tuple[float, float] = (10, 4)
                       ) -> plt.Figure:
        power = ds.cleaned_power if use_cleaned else ds.power_db_matrix
        freq_khz = ds.freq_axis_hz / 1e3

        fig, ax = plt.subplots(figsize=figsize)
        vmin = np.percentile(power[np.isfinite(power)], 2)
        vmax = np.percentile(power[np.isfinite(power)], 99.5)

        im = ax.imshow(
            power, aspect="auto", origin="lower", interpolation="nearest",
            extent=[ds.time_axis_s[0], ds.time_axis_s[-1],
                    freq_khz[0], freq_khz[-1]],
            cmap="inferno", vmin=vmin, vmax=vmax
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cbar.ax.tick_params(labelsize=6, colors=self.PALETTE["axis"])
        cbar.set_label("dBFS", fontsize=6, color=self.PALETTE["fg"])

        # Overlay RFI mask boundary
        if use_cleaned and ds.rfi_mask.any():
            rfi_channels = np.where(np.any(ds.rfi_mask, axis=1))[0]
            for ch in rfi_channels:
                ax.axhline(freq_khz[min(ch, len(freq_khz) - 1)],
                           color=self.PALETTE["danger"], lw=0.3, alpha=0.4)

        self._style_ax(ax, "WATERFALL — DYNAMIC SPECTRUM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq offset (kHz)")
        ax.text(0.01, 0.97,
                f"Δt={ds.time_resolution_s*1e3:.1f}ms  "
                f"Δf={ds.freq_resolution_hz:.0f}Hz  "
                f"RFI%={100*ds.rfi_mask.mean():.1f}",
                transform=ax.transAxes, color=self.PALETTE["accent"],
                fontsize=6, va="top")
        plt.tight_layout(pad=0.4)
        return fig

    # ─── Drift rate ───────────────────────────────────────────────────────────

    def plot_drift_track(self, ds: DynamicSpectrum,
                         drift_rate_hz_s: float,
                         figsize: Tuple[float, float] = (10, 3.5)
                         ) -> plt.Figure:
        power = ds.cleaned_power
        freq_khz = ds.freq_axis_hz / 1e3

        fig, ax = plt.subplots(figsize=figsize)
        vmin = np.percentile(power, 2)
        vmax = np.percentile(power, 99)
        ax.imshow(power, aspect="auto", origin="lower", interpolation="bilinear",
                  extent=[ds.time_axis_s[0], ds.time_axis_s[-1],
                          freq_khz[0], freq_khz[-1]],
                  cmap="plasma", vmin=vmin, vmax=vmax)

        # Overlay expected drift track
        t_center = (ds.time_axis_s[0] + ds.time_axis_s[-1]) / 2
        f_center = (freq_khz[0] + freq_khz[-1]) / 2
        t_track = ds.time_axis_s
        f_track_khz = f_center + drift_rate_hz_s * (t_track - t_center) / 1e3
        ax.plot(t_track, f_track_khz, color=self.PALETTE["accent"],
                lw=1.2, ls="--", alpha=0.8, label=f"Drift {drift_rate_hz_s:.3f} Hz/s")

        self._style_ax(ax, "DRIFT RATE ANALYSIS")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq offset (kHz)")
        ax.legend(fontsize=6, loc="upper right",
                  facecolor=self.PALETTE["bg"],
                  edgecolor=self.PALETTE["fg_dim"],
                  labelcolor=self.PALETTE["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    # ─── Pulsar profile ───────────────────────────────────────────────────────

    def plot_pulsar_profile(self, phase: np.ndarray, profile: np.ndarray,
                            fit: Optional[PulsarFit] = None,
                            figsize: Tuple[float, float] = (8, 3.5)
                            ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_profile, ax_residuals = axes

        # Profile
        ax_profile.fill_between(phase, profile, alpha=0.4, color=self.PALETTE["fg"])
        ax_profile.plot(phase, profile, color=self.PALETTE["fg"], lw=1.0)
        ax_profile.plot(np.concatenate([phase, phase + 1]),
                        np.concatenate([profile, profile]),
                        color=self.PALETTE["fg_dim"], lw=0.6, ls=":")
        if fit is not None:
            # Gaussian template overlay
            sigma = 0.05
            peak = phase[np.argmax(profile)]
            template = profile.max() * np.exp(-0.5 * ((phase - peak) / sigma) ** 2)
            ax_profile.plot(phase, template, color=self.PALETTE["accent"],
                            lw=0.8, ls="--", alpha=0.8, label="Gaussian fit")
            ax_profile.legend(fontsize=6, facecolor=self.PALETTE["bg"],
                              edgecolor=self.PALETTE["fg_dim"],
                              labelcolor=self.PALETTE["fg"])
        self._style_ax(ax_profile, "INTEGRATED PULSE PROFILE")
        ax_profile.set_xlabel("Phase")
        ax_profile.set_ylabel("Intensity (arb.)")

        # Timing residuals
        if fit is not None and len(fit.timing_residuals) > 0:
            t_bins = np.linspace(0, 1, len(fit.timing_residuals))
            ax_residuals.plot(t_bins, fit.timing_residuals, color=self.PALETTE["warn"],
                              lw=0.8)
            ax_residuals.axhline(0, color=self.PALETTE["fg_dim"], lw=0.6)
            ax_residuals.fill_between(t_bins, fit.timing_residuals,
                                      alpha=0.3, color=self.PALETTE["warn"])
            self._style_ax(ax_residuals,
                           f"TIMING RESIDUALS  χ²/dof={fit.chi2_reduced:.2f}")
            ax_residuals.set_xlabel("Phase")
            ax_residuals.set_ylabel("Residual (σ)")
        else:
            ax_residuals.text(0.5, 0.5, "NO FIT DATA", transform=ax_residuals.transAxes,
                              color=self.PALETTE["fg_dim"], ha="center", fontsize=8)
            self._style_ax(ax_residuals)

        plt.tight_layout(pad=0.4)
        return fig

    # ─── DM trial plot ────────────────────────────────────────────────────────

    def plot_dm_trials(self, dm_trials: np.ndarray, snr_vs_dm: np.ndarray,
                       figsize: Tuple[float, float] = (8, 3.0)
                       ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(dm_trials, snr_vs_dm, color=self.PALETTE["fg"],
                lw=0.9, alpha=0.9)
        ax.fill_between(dm_trials, snr_vs_dm,
                        np.full_like(snr_vs_dm, snr_vs_dm.min()),
                        alpha=0.15, color=self.PALETTE["fg"])
        best_idx = int(np.argmax(snr_vs_dm))
        ax.axvline(dm_trials[best_idx], color=self.PALETTE["accent"],
                   lw=1.0, ls="--", label=f"Best DM={dm_trials[best_idx]:.1f}")
        ax.scatter([dm_trials[best_idx]], [snr_vs_dm[best_idx]],
                   color=self.PALETTE["accent"], s=30, zorder=5)
        self._style_ax(ax, "DM TRIAL — INCOHERENT DE-DISPERSION")
        ax.set_xlabel("DM (pc·cm⁻³)")
        ax.set_ylabel("Peak SNR")
        ax.legend(fontsize=6, facecolor=self.PALETTE["bg"],
                  edgecolor=self.PALETTE["fg_dim"],
                  labelcolor=self.PALETTE["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    # ─── Spectral kurtosis ────────────────────────────────────────────────────

    def plot_spectral_kurtosis(self, freqs: np.ndarray, sk: np.ndarray,
                                figsize: Tuple[float, float] = (9, 3.0)
                                ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(freqs / 1e3, sk, color=self.PALETTE["fg"], lw=0.7)
        ax.axhline(1.0, color=self.PALETTE["warn"], lw=0.8, ls="--",
                   label="SK=1 (Gaussian noise)")
        ax.axhline(0.5, color=self.PALETTE["accent"], lw=0.6, ls=":",
                   label="SK=0.5 (CW/periodic)")
        # Flag RFI
        rfi_mask = np.abs(sk - 1.0) > 2.0
        if rfi_mask.any():
            ax.fill_between(freqs / 1e3, sk,
                            np.ones_like(sk), where=rfi_mask,
                            alpha=0.35, color=self.PALETTE["danger"],
                            label="RFI flagged")
        self._style_ax(ax, "SPECTRAL KURTOSIS")
        ax.set_xlabel("Freq offset (kHz)")
        ax.set_ylabel("SK")
        ax.set_ylim(-0.5, min(10, float(np.percentile(sk[np.isfinite(sk)], 98)) * 1.3))
        ax.legend(fontsize=6, facecolor=self.PALETTE["bg"],
                  edgecolor=self.PALETTE["fg_dim"],
                  labelcolor=self.PALETTE["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    # ─── SETI candidate summary ───────────────────────────────────────────────

    def plot_seti_scorecard(self, candidate: SETICandidate,
                            figsize: Tuple[float, float] = (8, 4.0)
                            ) -> plt.Figure:
        categories = [
            "SNR",
            "Water Hole",
            "Drift Rate",
            "DM Origin",
            "Coherence",
            "RFI Reject",
            "WoW Score",
        ]
        snr_score = min(10, max(0, candidate.snr_db / 4))
        wh_score = 10.0 if (WATER_HOLE[0] <= candidate.center_freq_mhz <= WATER_HOLE[1]) else \
                   max(0, 10 - min(abs(candidate.center_freq_mhz - WATER_HOLE[0]),
                                   abs(candidate.center_freq_mhz - WATER_HOLE[1])) / 30)
        drift_abs = abs(candidate.drift_rate_hz_s)
        drift_score = 10 if 0.001 < drift_abs < 0.5 else (6 if drift_abs < 2 else 1)
        dm_score = min(10, candidate.dispersion_measure / 20)
        coherence_score = candidate.coherence_score * 10
        rfi_score = candidate.rfi_rejection_passes * 2.5
        wow = candidate.wow_score

        values = [snr_score, wh_score, drift_score, dm_score,
                  coherence_score, rfi_score, wow]

        fig, (ax_radar, ax_bar) = plt.subplots(1, 2, figsize=figsize)

        # Radar chart
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vals = values + [values[0]]
        angles += angles[:1]
        ax_radar = plt.subplot(1, 2, 1, polar=True, facecolor=self.PALETTE["bg"])
        ax_radar.set_facecolor(self.PALETTE["bg"])
        ax_radar.plot(angles, vals, color=self.PALETTE["fg"], lw=1.2)
        ax_radar.fill(angles, vals, color=self.PALETTE["fg"], alpha=0.2)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, size=6, color=self.PALETTE["accent"])
        ax_radar.set_ylim(0, 10)
        ax_radar.set_yticks([2, 4, 6, 8, 10])
        ax_radar.set_yticklabels(["2", "4", "6", "8", "10"], size=5,
                                  color=self.PALETTE["fg_dim"])
        ax_radar.tick_params(colors=self.PALETTE["fg_dim"])
        ax_radar.spines["polar"].set_edgecolor(self.PALETTE["fg_dim"])
        ax_radar.grid(color=self.PALETTE["grid"], alpha=0.5)
        ax_radar.set_title(
            f"WoW={candidate.wow_score:.1f}  P(ET)={candidate.extraterrestrial_probability:.2e}",
            color=self.PALETTE["accent"], fontsize=7, pad=8)

        # Bar chart
        ax_b = plt.subplot(1, 2, 2)
        ax_b.set_facecolor(self.PALETTE["bg"])
        colors = [self.PALETTE["fg"] if v >= 5 else
                  self.PALETTE["warn"] if v >= 2 else
                  self.PALETTE["danger"] for v in values]
        bars = ax_b.barh(categories, values, color=colors, alpha=0.8, height=0.55)
        ax_b.set_xlim(0, 10)
        ax_b.axvline(7.0, color=self.PALETTE["accent"], lw=0.8, ls="--", alpha=0.6)
        for bar, val in zip(bars, values):
            ax_b.text(min(val + 0.1, 9.5), bar.get_y() + bar.get_height() / 2,
                      f"{val:.1f}", va="center", fontsize=5.5,
                      color=self.PALETTE["accent"])
        self._style_ax(ax_b, f"SETI SCORECARD — {candidate.classification}")
        ax_b.set_xlabel("Score (0–10)")
        for spine in ax_b.spines.values():
            spine.set_edgecolor(self.PALETTE["fg_dim"])
        ax_b.tick_params(colors=self.PALETTE["axis"])

        fig.set_facecolor(self.PALETTE["bg"])
        plt.tight_layout(pad=0.6)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def spectral_analyzer_page():
    from signal_engine import (
        init_session_state, SignalGenerator, SignalClass,
        _generate_for_class
    )
    init_session_state()

    st.markdown("""
    <style>
    .spec-header {
        font-family:'Courier New',monospace;
        color:#00ff88;
        font-size:0.78rem;
        letter-spacing:0.14em;
        border-bottom:1px solid #00ff4430;
        padding-bottom:0.4rem;
        margin-bottom:1rem;
    }
    .spec-label {
        font-family:'Courier New',monospace;
        color:#88ffcc;
        font-size:0.72rem;
        letter-spacing:0.08em;
        margin-top:0.6rem;
        margin-bottom:0.2rem;
    }
    .line-table {
        font-family:'Courier New',monospace;
        font-size:0.65rem;
        color:#00ff88;
    }
    .seti-alert {
        background:#001a00;
        border:1px solid #00ff88;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.72rem;
        color:#00ffcc;
        margin-bottom:0.5rem;
    }
    .danger-alert {
        background:#1a0000;
        border:1px solid #ff2222;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.72rem;
        color:#ff4444;
        margin-bottom:0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spec-header">[ SPECTRAL ANALYSIS MODULE — DEEP FREQUENCY DOMAIN INTELLIGENCE ]</div>',
                unsafe_allow_html=True)

    viz = SpectralVisualizer()
    gen: SignalGenerator = st.session_state.generator

    # ── Signal generation controls ─────────────────────────────────────────
    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown('<div class="spec-label">— INPUT SIGNAL —</div>', unsafe_allow_html=True)
        sig_type = st.selectbox(
            "Signal Class",
            [c.value for c in SignalClass if c != SignalClass.VOID_CARRIER],
        )
        center_mhz = st.number_input("Center Freq (MHz)", 100.0, 30000.0,
                                      1420.405751768, step=0.001, format="%.6f")
        duration   = st.slider("Duration (s)", 0.5, 30.0, 8.0, 0.5)
        snr_db     = st.slider("SNR (dB)", -5.0, 40.0, 18.0, 1.0)
        dm_val     = st.number_input("DM (pc·cm⁻³)", 0.0, 3000.0, 15.0, step=1.0)

        st.markdown('<div class="spec-label">— FFT CONFIG —</div>', unsafe_allow_html=True)
        n_fft     = st.select_slider("N_FFT", [128, 256, 512, 1024, 2048, 4096], 1024)
        wf_hop    = st.select_slider("Waterfall Hop", [32, 64, 128, 256], 128)
        window    = st.selectbox("Window", list(SpectralAnalyzer.WINDOW_REGISTRY.keys()), 3)
        excise    = st.checkbox("RFI Excision", value=True)

        st.markdown('<div class="spec-label">— ANALYSIS MODULES —</div>', unsafe_allow_html=True)
        do_pulsar  = st.checkbox("Pulsar Analysis", value=True)
        do_dm      = st.checkbox("DM Trial Search", value=True)
        do_doppler = st.checkbox("Doppler Analysis", value=True)
        do_seti    = st.checkbox("SETI Scoring", value=True)
        do_stokes  = st.checkbox("Polarimetry (Stokes)", value=False)

    with col_main:
        if st.button("▶ RUN SPECTRAL ANALYSIS", use_container_width=True):
            sig_class = SignalClass(sig_type)
            analyzer  = SpectralAnalyzer(sample_rate=2e6, center_freq_mhz=center_mhz)

            with st.spinner("[ PROCESSING SPECTRA ... ]"):
                t, signal = _generate_for_class(gen, sig_class, duration, snr_db, dm_val)
                # ── Power spectrum ─────────────────────────────────────────
                freqs, pdb = analyzer.calibrated_fft(signal, n_fft=n_fft, window=window)
                noise_floor, noise_sigma = analyzer.noise_power_density(pdb)
                lines = analyzer.detect_spectral_lines(freqs, pdb, n_sigma=4.0)

                # Averaged periodogram overlay
                freqs_avg, pdb_avg = analyzer.averaged_periodogram(
                    signal, n_fft=n_fft, n_overlap=n_fft // 2, window=window)

                fig_spectrum = viz.plot_power_spectrum(
                    freqs, pdb, lines=lines,
                    noise_floor=noise_floor,
                    title=f"POWER SPECTRUM — {sig_type} @ {center_mhz:.6f} MHz"
                )
                st.pyplot(fig_spectrum, use_container_width=True)
                plt.close(fig_spectrum)

                # ── Waterfall ──────────────────────────────────────────────
                ds = analyzer.build_dynamic_spectrum(
                    signal, n_fft=min(n_fft, 512), hop=wf_hop,
                    window=window, excise_rfi=excise)

                fig_wf = viz.plot_waterfall(ds, use_cleaned=excise)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

                # ── Spectral Kurtosis ──────────────────────────────────────
                sk_freqs, sk = analyzer.spectral_kurtosis(signal, n_fft=256)
                fig_sk = viz.plot_spectral_kurtosis(sk_freqs, sk)
                st.pyplot(fig_sk, use_container_width=True)
                plt.close(fig_sk)

                # ── Spectral lines table ───────────────────────────────────
                if lines:
                    st.markdown('<div class="spec-label">— DETECTED SPECTRAL LINES —</div>',
                                unsafe_allow_html=True)
                    line_data = []
                    for l in lines[:15]:
                        line_data.append({
                            "Freq (MHz)":  round(l.frequency_mhz, 6),
                            "Amp (dBFS)":  round(l.amplitude_db, 1),
                            "Width (Hz)":  round(l.width_hz, 1),
                            "Sig (σ)":     round(l.significance, 1),
                            "Vrad (km/s)": round(l.radial_vel_kms, 2),
                            "Match":       l.line_name or "—",
                            "RFI":         "⚠" if l.is_rfi else "✓",
                        })
                    st.dataframe(pd.DataFrame(line_data),
                                 use_container_width=True, hide_index=True)

                # ── Doppler Analysis ───────────────────────────────────────
                if do_doppler:
                    doppler = analyzer.compute_doppler_analysis(
                        signal, center_mhz, time_segments=min(32, int(duration * 2)))
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("Observed Freq",
                               f"{doppler.observed_freq_mhz:.6f} MHz",
                               delta=f"{doppler.doppler_shift_hz:+.1f} Hz")
                    dc2.metric("Radial Vel",    f"{doppler.radial_vel_kms:.2f} km/s")
                    dc3.metric("Drift Rate",    f"{doppler.drift_rate_hz_s:.4f} Hz/s")
                    dc4.metric("Barycentric Δf", f"{doppler.barycentric_corr_hz:.2f} Hz")

                    if doppler.acceleration_ms2 is not None:
                        st.markdown(
                            f'<div class="spec-label">Acceleration: '
                            f'{doppler.acceleration_ms2:.4f} m/s²</div>',
                            unsafe_allow_html=True)

                    fig_drift = viz.plot_drift_track(ds, doppler.drift_rate_hz_s)
                    st.pyplot(fig_drift, use_container_width=True)
                    plt.close(fig_drift)

                # ── Pulsar Analysis ────────────────────────────────────────
                if do_pulsar:
                    st.markdown('<div class="spec-label">— PULSAR / PERIODICITY SEARCH —</div>',
                                unsafe_allow_html=True)
                    candidates_p = analyzer.period_search_ffa(
                        signal, p_min_s=0.005, p_max_s=min(duration / 3, 5.0),
                        n_candidates=5)
                    if candidates_p:
                        best_p, best_snr_p = candidates_p[0]
                        fit = analyzer.fit_pulsar_candidate(signal, best_p, dm=dm_val)
                        phase, profile = analyzer.fold_pulsar(signal, fit.fitted_period_s)
                        fig_pulsar = viz.plot_pulsar_profile(phase, profile, fit)
                        st.pyplot(fig_pulsar, use_container_width=True)
                        plt.close(fig_pulsar)

                        ps1, ps2, ps3, ps4 = st.columns(4)
                        ps1.metric("Best Period", f"{fit.fitted_period_s:.6f} s",
                                   delta=f"±{fit.period_error_s:.2e} s")
                        ps2.metric("Profile SNR", f"{fit.profile_snr:.1f}")
                        ps3.metric("χ²/dof",      f"{fit.chi2_reduced:.2f}")
                        ps4.metric("Catalog Match", fit.matched_catalog or "NONE")

                        # Period candidates table
                        cand_rows = [{"Period (s)": round(p, 6), "SNR": round(s, 1)}
                                     for p, s in candidates_p]
                        st.dataframe(pd.DataFrame(cand_rows),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.markdown('<div class="spec-label">NO SIGNIFICANT PERIODICITY DETECTED.</div>',
                                    unsafe_allow_html=True)

                # ── DM Trial ───────────────────────────────────────────────
                if do_dm:
                    st.markdown('<div class="spec-label">— INCOHERENT DE-DISPERSION —</div>',
                                unsafe_allow_html=True)
                    dm_trials = np.linspace(0, min(500, dm_val * 5 + 100), 150)
                    dm_trials_out, snr_dm = analyzer.incoherent_dedispersion(ds, dm_trials)
                    fig_dm = viz.plot_dm_trials(dm_trials_out, snr_dm)
                    st.pyplot(fig_dm, use_container_width=True)
                    plt.close(fig_dm)
                    best_dm = float(dm_trials_out[np.argmax(snr_dm)])
                    st.markdown(
                        f'<div class="spec-label">BEST-FIT DM: {best_dm:.1f} pc·cm⁻³ | '
                        f'INJECTED DM: {dm_val:.1f} pc·cm⁻³</div>',
                        unsafe_allow_html=True)

                # ── SETI Scoring ───────────────────────────────────────────
                if do_seti:
                    st.markdown('<div class="spec-label">— SETI CANDIDATE ASSESSMENT —</div>',
                                unsafe_allow_html=True)
                    meta = {
                        "uid": "LIVE",
                        "center_freq_mhz": center_mhz,
                        "dispersion_measure": dm_val,
                    }
                    candidate = analyzer.score_seti_candidate(signal, meta)
                    fig_seti = viz.plot_seti_scorecard(candidate)
                    st.pyplot(fig_seti, use_container_width=True)
                    plt.close(fig_seti)

                    alert_class = "seti-alert" if candidate.wow_score < 7 else "danger-alert"
                    st.markdown(
                        f'<div class="{alert_class}">'
                        f'CLASSIFICATION: {candidate.classification} | '
                        f'WoW={candidate.wow_score:.2f}/10 | '
                        f'P(ET)={candidate.extraterrestrial_probability:.3e} | '
                        f'RFI-REJECT PASSES: {candidate.rfi_rejection_passes}/4<br>'
                        f'{candidate.notes if candidate.notes else "NO ANOMALOUS MARKERS"}'
                        f'</div>',
                        unsafe_allow_html=True)

                # ── Stokes Polarimetry ─────────────────────────────────────
                if do_stokes:
                    st.markdown('<div class="spec-label">— STOKES PARAMETERS (SIMULATED DUAL-POL) —</div>',
                                unsafe_allow_html=True)
                    # Simulate second polarization with slight rotation
                    rng = np.random.default_rng(77)
                    signal_y = (signal * np.exp(1j * rng.uniform(0, 0.3, len(signal))) +
                                0.1 * (rng.normal(0, 1, len(signal)) +
                                       1j * rng.normal(0, 1, len(signal))))
                    I, Q, U, V = analyzer.compute_stokes(signal, signal_y)
                    lin_frac, circ_frac, tot_frac = analyzer.polarization_fraction(I, Q, U, V)
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Linear Pol", f"{lin_frac * 100:.1f}%")
                    sc2.metric("Circular Pol", f"{circ_frac * 100:.1f}%")
                    sc3.metric("Total Pol", f"{tot_frac * 100:.1f}%")

                    fig_stokes, axes = plt.subplots(4, 1, figsize=(9, 6),
                                                     facecolor=viz.PALETTE["bg"],
                                                     sharex=True)
                    t_axis = t[:len(I)]
                    stokes_data = [(I, "I", viz.PALETTE["fg"]),
                                   (Q, "Q", "#ff8800"),
                                   (U, "U", "#0088ff"),
                                   (V, "V", "#ff44ff")]
                    for ax, (data, label, color) in zip(axes, stokes_data):
                        ax.plot(t_axis[:min(len(data), 2000)],
                                data[:min(len(data), 2000)],
                                color=color, lw=0.6, alpha=0.85)
                        ax.set_facecolor(viz.PALETTE["bg"])
                        ax.set_ylabel(f"Stokes {label}", fontsize=6,
                                      color=color)
                        for sp in ax.spines.values():
                            sp.set_edgecolor(viz.PALETTE["fg_dim"])
                        ax.tick_params(colors=viz.PALETTE["axis"])
                    axes[-1].set_xlabel("Time (s)", fontsize=6,
                                        color=viz.PALETTE["fg"])
                    axes[0].set_title("STOKES PARAMETER TIME SERIES",
                                      color=viz.PALETTE["fg"], fontsize=7, loc="left")
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig_stokes, use_container_width=True)
                    plt.close(fig_stokes)

    # ── Spectral line catalog reference ────────────────────────────────────
    with st.expander("[ SPECTRAL LINE CATALOG ]"):
        catalog_rows = [
            {"Line": name, "Rest Freq (MHz)": round(freq, 6),
             "In Water Hole": "✓" if WATER_HOLE[0] <= freq <= WATER_HOLE[1] else "—",
             "Δv per MHz (km/s)": round(SPEED_OF_LIGHT_MS / 1e3 / freq, 2)}
            for name, freq in SPECTRAL_LINE_CATALOG.items()
        ]
        st.dataframe(pd.DataFrame(catalog_rows),
                     use_container_width=True, hide_index=True)

    with st.expander("[ KNOWN PULSAR CATALOG ]"):
        pulsar_rows = [
            {"Name": n, "Period (s)": round(p, 7), "DM (pc·cm⁻³)": round(d, 1),
             "Spin-down age (kyr)": round(p / (2 * 1e-15 * 1e3), 0) if p > 0.01 else "—"}
            for n, p, d in KNOWN_PULSARS
        ]
        st.dataframe(pd.DataFrame(pulsar_rows),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    spectral_analyzer_page()
