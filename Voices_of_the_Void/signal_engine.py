"""
signal_engine.py — Alpen Signal Observatorium Core Processing Engine
Voices of the Void | Array Operations System v2.4.1
Dr. Kel Station — Dunkeltaler Forest, Swiss Alps

Handles: signal taxonomy, physical generation models, DSP pipeline,
         ML classification, drive management, real-time acquisition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.fft as sp_fft
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SPEED_OF_LIGHT_MS   = 2.99792458e8          # m/s
BOLTZMANN_K         = 1.380649e-23           # J/K
PLANCK_H            = 6.62607015e-34         # J·s
HYDROGEN_LINE_MHZ   = 1420.405751768         # MHz — 21cm H-I line
WATER_HOLE_LOW_MHZ  = 1420.405751768         # MHz
WATER_HOLE_HIGH_MHZ = 1666.0                 # MHz — OH radical
PARSEC_M            = 3.085677581e16         # m
DISPERSION_CONST    = 4.148808e3             # DM constant (MHz² pc cm³ s⁻¹)
SYSTEM_TEMP_K       = 25.0                   # Typical cryogenic receiver temp
SKY_TEMP_K          = 3.5                    # Galactic background at 1.4 GHz
EFFECTIVE_AREA_M2   = 75.0                   # Effective collecting area (m²)
BANDWIDTH_HZ        = 1e6                    # Default channel bandwidth

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TAXONOMY — Mirroring VotV classification schema
# ─────────────────────────────────────────────────────────────────────────────

class SignalClass(Enum):
    NARROWBAND_CW       = "NARROWBAND_CW"       # Continuous wave, narrow BW
    NARROWBAND_PULSED   = "NARROWBAND_PULSED"   # Pulsed narrowband
    PULSAR              = "PULSAR"              # Rotating neutron star
    CHIRP               = "CHIRP"              # Linear freq sweep
    BROADBAND_BURST     = "BROADBAND_BURST"    # Fast radio burst type
    STRUCTURED_BPSK     = "STRUCTURED_BPSK"    # Binary phase-shift keying
    STRUCTURED_FSK      = "STRUCTURED_FSK"     # Frequency-shift keying
    ASTROPHYSICAL_LINE  = "ASTROPHYSICAL_LINE"  # Spectral emission/absorption
    ANOMALOUS           = "ANOMALOUS"          # Unknown — flagged for review
    ARIRAL              = "ARIRAL"             # Classified: non-human origin
    VOID_CARRIER        = "VOID_CARRIER"       # The End Is Near class — RESTRICTED

class ThreatLevel(Enum):
    NOMINAL      = 0
    ELEVATED     = 1
    CRITICAL     = 2
    CONTAINMENT  = 3

class SignalOrigin(Enum):
    SOLAR_SYSTEM    = "SOLAR_SYSTEM"
    GALACTIC        = "GALACTIC"
    EXTRAGALACTIC   = "EXTRAGALACTIC"
    UNKNOWN         = "UNKNOWN"
    CLASSIFIED      = "CLASSIFIED"

class ProcessingStage(Enum):
    RAW         = 0   # Just captured
    FILTERED    = 1   # Band-pass filtered
    REFINED     = 2   # Denoised + parameterized
    CLASSIFIED  = 3   # ML label assigned
    ARCHIVED    = 4   # Written to drive, ready for HQ

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DATACLASS — complete parameter record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    uid:                str             = field(default_factory=lambda: str(uuid.uuid4())[:12].upper())
    timestamp:          float           = field(default_factory=time.time)
    signal_class:       SignalClass     = SignalClass.NARROWBAND_CW
    origin:             SignalOrigin    = SignalOrigin.UNKNOWN
    threat_level:       ThreatLevel    = ThreatLevel.NOMINAL
    stage:              ProcessingStage = ProcessingStage.RAW

    # Frequency domain
    center_freq_mhz:    float = 1420.0
    bandwidth_hz:       float = 1000.0
    drift_rate_hz_s:    float = 0.0         # Hz/s — Doppler drift

    # Time domain
    duration_s:         float = 10.0
    sample_rate_hz:     float = 2e6
    period_s:           Optional[float] = None   # For pulsars / pulsed

    # Radiometric
    snr_db:             float = 0.0
    flux_density_jy:    float = 0.0         # Janskys
    dispersion_measure: float = 0.0         # pc cm⁻³

    # Spatial
    ra_hours:           float = 0.0         # Right ascension
    dec_degrees:        float = 0.0         # Declination
    distance_kpc:       Optional[float] = None

    # Processing metadata
    classifier_confidence: float  = 0.0
    anomaly_score:         float  = 0.0
    wow_factor:            float  = 0.0     # SETI significance score
    notes:                 str    = ""
    hash_code:             str    = field(default_factory=lambda: hashlib.sha256(
                                        str(time.time_ns()).encode()).hexdigest()[:16].upper())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["signal_class"]  = self.signal_class.value
        d["origin"]        = self.origin.value
        d["threat_level"]  = self.threat_level.value
        d["stage"]         = self.stage.value
        return d

    def to_dataframe_row(self) -> Dict[str, Any]:
        return {
            "UID":          self.uid,
            "Timestamp":    pd.Timestamp(self.timestamp, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "Class":        self.signal_class.value,
            "Origin":       self.origin.value,
            "Threat":       self.threat_level.name,
            "Stage":        self.stage.name,
            "Freq(MHz)":    round(self.center_freq_mhz, 6),
            "BW(Hz)":       int(self.bandwidth_hz),
            "Drift(Hz/s)":  round(self.drift_rate_hz_s, 4),
            "SNR(dB)":      round(self.snr_db, 2),
            "Flux(Jy)":     round(self.flux_density_jy, 4),
            "DM":           round(self.dispersion_measure, 2),
            "RA(h)":        round(self.ra_hours, 4),
            "Dec(°)":       round(self.dec_degrees, 4),
            "WoW":          round(self.wow_factor, 3),
            "Conf%":        round(self.classifier_confidence * 100, 1),
            "Hash":         self.hash_code,
        }

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Generates physically realistic synthetic baseband signals
    for each signal class, with proper noise, dispersion, and modulation.
    """

    def __init__(self, sample_rate: float = 2e6, seed: Optional[int] = None):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def _time_axis(self, duration: float) -> np.ndarray:
        n_samples = int(duration * self.sample_rate)
        return np.linspace(0, duration, n_samples, endpoint=False)

    def _awgn(self, n: int, sigma: float) -> np.ndarray:
        """Additive White Gaussian Noise — complex baseband."""
        return (self.rng.normal(0, sigma, n) +
                1j * self.rng.normal(0, sigma, n)) / np.sqrt(2)

    def _thermal_noise_sigma(self, bandwidth: float) -> float:
        """
        Compute noise standard deviation from radiometer equation:
        T_sys = T_rx + T_sky
        sigma = sqrt(k * T_sys * B)
        Returns normalized amplitude for unit signal power reference.
        """
        t_sys = SYSTEM_TEMP_K + SKY_TEMP_K
        noise_power = BOLTZMANN_K * t_sys * bandwidth
        return np.sqrt(noise_power * 1e26)  # scale to Jansky domain

    def _apply_dispersion(self, signal: np.ndarray, dm: float,
                          freq_mhz: float, bandwidth_hz: float) -> np.ndarray:
        """
        Apply interstellar dispersion smearing via frequency-dependent delay.
        t_delay = DM * k_DM * (1/f_lo² − 1/f_hi²) seconds
        Implemented as a phase ramp in the frequency domain.
        """
        if dm == 0:
            return signal
        n = len(signal)
        freqs = sp_fft.fftfreq(n, d=1.0 / self.sample_rate)  # baseband freqs (Hz)
        f_center = freq_mhz * 1e6
        # Dispersion delay in seconds relative to center channel
        k = DISPERSION_CONST * dm
        delays = k / ((f_center + freqs) ** 2 / 1e12) - k / (f_center ** 2 / 1e12)
        phase_ramp = np.exp(-2j * np.pi * freqs * delays)
        S = sp_fft.fft(signal)
        return sp_fft.ifft(S * phase_ramp)

    def generate_narrowband_cw(self, duration: float, freq_offset_hz: float,
                               snr_db: float, bandwidth: float = 100.0,
                               dm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous wave narrowband signal — the most common SETI candidate."""
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        # Pure sinusoid at offset frequency
        cw = signal_amp * np.exp(2j * np.pi * freq_offset_hz * t)
        # Amplitude fluctuation (scintillation)
        scintillation = 1 + 0.05 * np.sin(2 * np.pi * 0.1 * t) + \
                        0.02 * self.rng.normal(0, 1, n)
        cw *= scintillation
        noise = self._awgn(n, 1.0)
        result = cw + noise
        if dm > 0:
            result = self._apply_dispersion(result, dm, 1420.0, bandwidth)
        return t, result

    def generate_narrowband_pulsed(self, duration: float, freq_offset_hz: float,
                                   snr_db: float, pulse_period: float,
                                   pulse_width: float, dm: float = 0.0
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Pulsed narrowband — periodic bursts at fixed frequency."""
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        envelope = np.zeros(n)
        pulse_samples = int(pulse_width * self.sample_rate)
        period_samples = int(pulse_period * self.sample_rate)
        for start in range(0, n, period_samples):
            end = min(start + pulse_samples, n)
            envelope[start:end] = 1.0
        # Apply raised-cosine shaping to avoid spectral splatter
        rc_len = min(64, pulse_samples // 4)
        if rc_len > 0:
            rc = 0.5 * (1 - np.cos(np.pi * np.arange(rc_len) / rc_len))
            kernel = np.concatenate([rc, np.ones(max(0, pulse_samples - 2*rc_len)), rc[::-1]])
            if len(kernel) > 0 and len(envelope) > len(kernel):
                envelope = np.convolve(envelope, kernel / kernel.sum(), mode="same")
        carrier = signal_amp * envelope * np.exp(2j * np.pi * freq_offset_hz * t)
        noise = self._awgn(n, 1.0)
        result = carrier + noise
        if dm > 0:
            result = self._apply_dispersion(result, dm, 1420.0, 1e4)
        return t, result

    def generate_pulsar(self, duration: float, period_s: float,
                        snr_db: float, dm: float,
                        profile_type: str = "gaussian"
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physically realistic pulsar signal.
        Profile types: 'gaussian', 'double_peaked', 'exponential'
        Includes timing noise and pulse-to-pulse intensity variations (giant pulses).
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        phase = (t % period_s) / period_s

        if profile_type == "gaussian":
            sigma = 0.05
            profile = np.exp(-0.5 * ((phase - 0.5) / sigma) ** 2)
        elif profile_type == "double_peaked":
            sigma = 0.03
            profile = (np.exp(-0.5 * ((phase - 0.35) / sigma) ** 2) +
                       0.7 * np.exp(-0.5 * ((phase - 0.65) / sigma) ** 2))
        else:  # exponential decay
            profile = np.where(phase > 0.45,
                               np.exp(-10 * (phase - 0.45)) * (phase < 0.75),
                               0.0)

        # Giant pulse events (rare intensity spikes)
        giant_mask = self.rng.random(n) < 0.001
        giant_amplitude = self.rng.exponential(5.0, n) * giant_mask
        envelope = signal_amp * (profile + giant_amplitude)

        # Timing noise: stochastic phase wander
        timing_noise = np.cumsum(self.rng.normal(0, 1e-5, n)) * self.sample_rate / n
        carrier_freq = HYDROGEN_LINE_MHZ * 0.1e6  # baseband offset
        carrier = envelope * np.exp(2j * np.pi * (carrier_freq * t + timing_noise))

        noise = self._awgn(n, 1.0)
        result = carrier + noise
        result = self._apply_dispersion(result, dm, 1400.0, 1e6)
        return t, result

    def generate_chirp(self, duration: float, snr_db: float,
                       f_start_hz: float, f_end_hz: float,
                       dm: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linear frequency chirp — could indicate planetary radar, FRB echo, or structured transmission.
        Uses quadratic phase for exact linear sweep.
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        chirp_rate = (f_end_hz - f_start_hz) / duration
        phase = 2 * np.pi * (f_start_hz * t + 0.5 * chirp_rate * t ** 2)
        signal = signal_amp * np.exp(1j * phase)
        # Amplitude envelope (Gaussian taper to avoid edge artifacts)
        taper = np.exp(-0.5 * ((t - duration / 2) / (duration / 4)) ** 2)
        signal *= taper
        noise = self._awgn(n, 1.0)
        result = signal + noise
        if dm > 0:
            result = self._apply_dispersion(result, dm, 1400.0, abs(f_end_hz - f_start_hz))
        return t, result

    def generate_fast_radio_burst(self, duration: float, snr_db: float,
                                  dm: float, burst_width_ms: float = 5.0
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Radio Burst — astrophysically realistic.
        Broadband, millisecond duration, high DM, potential repeater.
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0) * 10  # FRBs are bright
        burst_center = duration / 2
        burst_sigma = (burst_width_ms * 1e-3) / 2.355  # FWHM to sigma

        # Intrinsic burst profile
        envelope = signal_amp * np.exp(-0.5 * ((t - burst_center) / burst_sigma) ** 2)

        # Scattering tail (exponential scatter broadening)
        scattering_time = 1e-3 * (dm / 100) ** 2
        if scattering_time > 1e-6:
            scatter_kernel_len = min(int(5 * scattering_time * self.sample_rate), n // 4)
            if scatter_kernel_len > 1:
                scatter_t = np.arange(scatter_kernel_len) / self.sample_rate
                scatter_kernel = np.exp(-scatter_t / scattering_time)
                scatter_kernel /= scatter_kernel.sum()
                envelope = np.convolve(envelope, scatter_kernel, mode="same")

        # Broadband noise-like waveform (FRBs are not coherent at baseband)
        waveform = envelope * (self.rng.normal(0, 1, n) + 1j * self.rng.normal(0, 1, n))
        noise = self._awgn(n, 1.0)
        result = waveform + noise
        result = self._apply_dispersion(result, dm, 1400.0, 1e8)
        return t, result

    def generate_bpsk(self, duration: float, snr_db: float,
                      bit_rate_bps: float, carrier_offset_hz: float,
                      preamble: Optional[bytes] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Binary Phase-Shift Keying — structured data-bearing signal.
        Raises threat level; may require cryptographic analysis.
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        samples_per_bit = int(self.sample_rate / bit_rate_bps)
        n_bits = n // samples_per_bit

        if preamble is not None:
            bits = np.unpackbits(np.frombuffer(preamble, dtype=np.uint8))[:n_bits]
            if len(bits) < n_bits:
                bits = np.concatenate([bits, self.rng.integers(0, 2, n_bits - len(bits))])
        else:
            bits = self.rng.integers(0, 2, n_bits)

        # Map bits to phases: 0→0°, 1→180°
        symbols = 1 - 2 * bits.astype(float)
        # Upsample and apply root-raised-cosine filter
        symbols_upsampled = np.repeat(symbols, samples_per_bit)[:n]
        # Simple rectangular pulse here; RRC would be more accurate
        signal = signal_amp * symbols_upsampled * np.exp(2j * np.pi * carrier_offset_hz * t)
        noise = self._awgn(n, 1.0)
        return t, signal + noise

    def generate_fsk(self, duration: float, snr_db: float,
                     bit_rate_bps: float, freq1_hz: float,
                     freq2_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency-Shift Keying — common in telemetry, possible interstellar beacon.
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        samples_per_bit = int(self.sample_rate / bit_rate_bps)
        n_bits = n // samples_per_bit
        bits = self.rng.integers(0, 2, n_bits)
        bits_upsampled = np.repeat(bits, samples_per_bit)[:n]
        freq_inst = np.where(bits_upsampled == 0, freq1_hz, freq2_hz)
        phase = 2 * np.pi * np.cumsum(freq_inst) / self.sample_rate
        signal = signal_amp * np.exp(1j * phase)
        noise = self._awgn(n, 1.0)
        return t, signal + noise

    def generate_anomalous(self, duration: float, snr_db: float
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Anomalous signal — structured but unclassified.
        Non-repeating, non-random, fractal-ish structure.
        Inspired by VotV 'ARIRAL' class signals.
        """
        t = self._time_axis(duration)
        n = len(t)
        signal_amp = 10 ** (snr_db / 20.0)
        # Fractal frequency modulation via Weierstrass function
        fm_phase = np.zeros(n)
        for k in range(1, 12):
            a = 0.5 ** k
            b = 3 ** k
            fm_phase += a * np.sin(2 * np.pi * b * 0.01 * t)
        fm_phase *= 200 * 2 * np.pi  # scale to meaningful freq deviation

        # Non-stationary amplitude envelope
        am_envelope = np.abs(np.sin(2 * np.pi * 0.07 * t)) ** 0.3
        am_envelope *= (1 + 0.3 * np.cos(2 * np.pi * 0.23 * t))

        signal = signal_amp * am_envelope * np.exp(1j * fm_phase)
        # Correlated noise (not AWGN — suspicious)
        corr_noise = np.convolve(self._awgn(n + 100, 0.3),
                                 np.exp(-np.arange(100) / 10), mode="valid")[:n]
        return t, signal + corr_noise

    def generate_void_carrier(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        RESTRICTED — Class VOID_CARRIER
        The End Is Near signal archetype.
        DO NOT archive. DO NOT transmit to HQ. Erase drive immediately.
        """
        t = self._time_axis(duration)
        n = len(t)
        # Prime-number frequency grid — unmistakably artificial
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        signal = np.zeros(n, dtype=complex)
        for i, p in enumerate(primes):
            freq = p * 1000.0  # prime multiples of 1 kHz
            amp = 1.0 / np.sqrt(i + 1)
            signal += amp * np.exp(2j * np.pi * freq * t)
        # Embedded binary sequence — Fibonacci word
        fib_word = self._fibonacci_word(min(n, 4096))
        fib_upsampled = np.repeat(fib_word, max(1, n // len(fib_word)))[:n]
        modulation = 1.0 + 0.15 * (fib_upsampled * 2 - 1)
        signal *= modulation[:n]
        # No noise — suspiciously clean
        return t, signal / np.max(np.abs(signal)) * 2.0

    @staticmethod
    def _fibonacci_word(length: int) -> np.ndarray:
        a, b = "1", "10"
        while len(b) < length:
            a, b = b, b + a
        return np.array([int(c) for c in b[:length]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# DSP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DSPEngine:
    """
    Digital Signal Processing pipeline.
    All operations maintain full numerical precision.
    """

    WINDOW_FUNCTIONS = {
        "hamming":    np.hamming,
        "hann":       np.hanning,
        "blackman":   np.blackman,
        "blackman_harris": lambda n: sp_signal.windows.blackmanharris(n),
        "kaiser_8":   lambda n: np.kaiser(n, 8.0),
        "kaiser_14":  lambda n: np.kaiser(n, 14.0),
        "flattop":    lambda n: sp_signal.windows.flattop(n),
        "rectangular": np.ones,
    }

    def __init__(self, sample_rate: float = 2e6):
        self.sample_rate = sample_rate

    def compute_fft(self, signal: np.ndarray, window: str = "blackman_harris",
                    n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute windowed FFT with proper normalization.
        Returns (frequency_axis_hz, power_spectrum_db).
        """
        n = len(signal)
        if n_fft is None:
            n_fft = n
        win_fn = self.WINDOW_FUNCTIONS.get(window, np.hanning)
        win = win_fn(n)
        # Coherent gain correction
        cgain = np.mean(win)
        windowed = signal[:n] * win
        spectrum = sp_fft.fft(windowed, n=n_fft)
        freqs = sp_fft.fftfreq(n_fft, d=1.0 / self.sample_rate)
        # One-sided spectrum
        half = n_fft // 2
        freqs_pos = freqs[:half]
        power = (np.abs(spectrum[:half]) / (n * cgain)) ** 2
        power_db = 10 * np.log10(power + 1e-20)
        return freqs_pos, power_db

    def compute_stft(self, signal: np.ndarray, n_fft: int = 512,
                     hop: int = 128, window: str = "hann"
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Short-Time Fourier Transform for waterfall / spectrogram.
        Returns (time_bins, frequency_bins, power_dB_matrix).
        """
        win_fn = self.WINDOW_FUNCTIONS.get(window, np.hanning)
        win = win_fn(n_fft)
        n = len(signal)
        n_frames = (n - n_fft) // hop + 1
        stft_matrix = np.zeros((n_fft // 2, n_frames), dtype=complex)
        for i in range(n_frames):
            frame = signal[i * hop: i * hop + n_fft] * win
            spec = sp_fft.fft(frame, n=n_fft)
            stft_matrix[:, i] = spec[:n_fft // 2]
        freqs = sp_fft.fftfreq(n_fft, d=1.0 / self.sample_rate)[:n_fft // 2]
        times = np.arange(n_frames) * hop / self.sample_rate
        power_db = 10 * np.log10(np.abs(stft_matrix) ** 2 + 1e-20)
        return times, freqs, power_db

    def design_bandpass_filter(self, low_hz: float, high_hz: float,
                               order: int = 8, filter_type: str = "butter"
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design IIR bandpass filter using scipy.
        Returns (b, a) coefficients for use with sosfilt.
        """
        nyq = self.sample_rate / 2
        low_n = max(1e-4, low_hz / nyq)
        high_n = min(0.9999, high_hz / nyq)
        if filter_type == "butter":
            sos = sp_signal.butter(order, [low_n, high_n], btype="bandpass", output="sos")
        elif filter_type == "cheby1":
            sos = sp_signal.cheby1(order, 0.5, [low_n, high_n], btype="bandpass", output="sos")
        elif filter_type == "ellip":
            sos = sp_signal.ellip(order, 0.5, 60, [low_n, high_n], btype="bandpass", output="sos")
        else:
            sos = sp_signal.butter(order, [low_n, high_n], btype="bandpass", output="sos")
        return sos

    def apply_bandpass(self, signal: np.ndarray, low_hz: float, high_hz: float,
                       order: int = 8, filter_type: str = "butter") -> np.ndarray:
        """Apply zero-phase bandpass filter."""
        sos = self.design_bandpass_filter(low_hz, high_hz, order, filter_type)
        if np.iscomplexobj(signal):
            return (sp_signal.sosfiltfilt(sos, signal.real) +
                    1j * sp_signal.sosfiltfilt(sos, signal.imag))
        return sp_signal.sosfiltfilt(sos, signal)

    def matched_filter(self, signal: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        Matched filter (cross-correlation) for maximum SNR detection.
        Returns normalized cross-correlation output.
        """
        mf_output = sp_signal.correlate(signal, template, mode="full")
        norm = np.sqrt(np.sum(np.abs(signal) ** 2) * np.sum(np.abs(template) ** 2))
        return mf_output / (norm + 1e-20)

    def estimate_snr(self, signal: np.ndarray, signal_band: Tuple[float, float],
                     noise_bands: List[Tuple[float, float]]) -> float:
        """
        Estimate SNR using on-off band technique.
        signal_band: (low_hz, high_hz) containing the signal
        noise_bands: list of (low_hz, high_hz) for off-signal estimation
        """
        freqs, power_db = self.compute_fft(signal, window="blackman_harris")
        power_linear = 10 ** (power_db / 10)

        # Signal power
        sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        sig_power = np.mean(power_linear[sig_mask]) if sig_mask.any() else 0

        # Noise power
        noise_power_vals = []
        for lo, hi in noise_bands:
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.any():
                noise_power_vals.append(np.mean(power_linear[mask]))
        noise_power = np.mean(noise_power_vals) if noise_power_vals else 1e-20

        snr_linear = (sig_power - noise_power) / (noise_power + 1e-20)
        return 10 * np.log10(max(snr_linear, 1e-10))

    def detect_drift_rate(self, signal: np.ndarray, time_segments: int = 16
                          ) -> Tuple[float, np.ndarray]:
        """
        Estimate Doppler drift rate via linear regression on peak frequency vs. time.
        Returns (drift_rate_hz_s, peak_freq_vs_time_array).
        """
        n = len(signal)
        seg_len = n // time_segments
        peak_freqs = []
        for i in range(time_segments):
            seg = signal[i * seg_len:(i + 1) * seg_len]
            freqs, pdb = self.compute_fft(seg, window="hann")
            if len(pdb) > 0:
                peak_freqs.append(freqs[np.argmax(pdb)])
        if len(peak_freqs) < 2:
            return 0.0, np.array(peak_freqs)
        seg_duration = n / (self.sample_rate * time_segments)
        times = np.arange(len(peak_freqs)) * seg_duration
        coeffs = np.polyfit(times, peak_freqs, 1)
        return float(coeffs[0]), np.array(peak_freqs)

    def autocorrelation(self, signal: np.ndarray, max_lag: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Full normalized autocorrelation for periodicity detection."""
        n = len(signal)
        power = np.abs(signal) ** 2
        if max_lag is None:
            max_lag = n // 2
        ac = sp_signal.correlate(power, power, mode="full")
        ac = ac[n - 1:]  # Keep positive lags
        ac = ac[:max_lag] / (ac[0] + 1e-20)
        lag_times = np.arange(max_lag) / self.sample_rate
        return lag_times, ac

    def detect_periodicity(self, signal: np.ndarray, min_period_s: float = 0.001,
                           max_period_s: float = 10.0) -> Optional[float]:
        """
        Detect periodicity using Lomb-Scargle periodogram on envelope.
        Returns best-fit period in seconds, or None if no significant peak.
        """
        from scipy.signal import lombscargle
        n = len(signal)
        t = np.arange(n) / self.sample_rate
        envelope = np.abs(signal)
        envelope -= envelope.mean()
        min_freq = 1.0 / max_period_s
        max_freq = 1.0 / min_period_s
        n_freqs = 2000
        ang_freqs = 2 * np.pi * np.linspace(min_freq, max_freq, n_freqs)
        try:
            pgram = lombscargle(t, envelope, ang_freqs, normalize=True)
            peak_idx = np.argmax(pgram)
            if pgram[peak_idx] > 0.6:  # Significance threshold
                best_ang_freq = ang_freqs[peak_idx]
                return float(2 * np.pi / best_ang_freq)
        except Exception:
            pass
        return None

    def extract_features(self, signal: np.ndarray, freq_mhz: float = 1420.0
                         ) -> Dict[str, float]:
        """
        Extract comprehensive feature vector for ML classification.
        33-dimensional feature vector.
        """
        n = len(signal)
        real_part = signal.real
        imag_part = signal.imag
        env = np.abs(signal)
        phase = np.angle(signal)
        phase_diff = np.diff(np.unwrap(phase))

        freqs, pdb = self.compute_fft(signal, window="blackman_harris")
        power_lin = 10 ** (pdb / 10)

        # Peak frequency stats
        peak_idx = np.argmax(pdb)
        peak_freq = float(freqs[peak_idx]) if len(freqs) > 0 else 0.0

        # Bandwidth (10dB below peak)
        peak_val = pdb[peak_idx] if len(pdb) > 0 else -200
        bw_mask = pdb > (peak_val - 10)
        bw_10db = float(np.sum(bw_mask) * (freqs[1] - freqs[0])) if (len(freqs) > 1 and bw_mask.any()) else 0.0

        # Spectral entropy
        if power_lin.sum() > 0:
            pnorm = power_lin / power_lin.sum()
            spec_entropy = float(scipy_entropy(pnorm + 1e-20))
        else:
            spec_entropy = 0.0

        # Spectral flatness (Wiener entropy)
        geo_mean = float(np.exp(np.mean(np.log(power_lin + 1e-20))))
        arith_mean = float(np.mean(power_lin))
        spec_flatness = geo_mean / (arith_mean + 1e-20)

        # Time-domain stats
        mean_env = float(np.mean(env))
        std_env = float(np.std(env))
        kurt_env = float(kurtosis(env))
        skew_env = float(skew(env))

        # Phase stats
        mean_phase_diff = float(np.mean(np.abs(phase_diff))) if len(phase_diff) > 0 else 0.0
        std_phase_diff = float(np.std(phase_diff)) if len(phase_diff) > 0 else 0.0

        # Autocorrelation peak at lag > 0
        lag_t, ac = self.autocorrelation(signal, max_lag=min(n // 4, 10000))
        second_peak_ac = float(np.max(ac[10:])) if len(ac) > 10 else 0.0
        periodicity_score = float(np.mean(ac[1:] > 0.5)) if len(ac) > 1 else 0.0

        # Frequency drift
        drift_rate, _ = self.detect_drift_rate(signal)

        # Crest factor
        crest_factor = float(np.max(env) / (mean_env + 1e-10))

        # AM/FM indices
        am_index = std_env / (mean_env + 1e-10)
        fm_index = std_phase_diff / (mean_phase_diff + 1e-10)

        # Zero crossing rate
        zcr = float(np.mean(np.diff(np.sign(real_part)) != 0))

        # Instantaneous frequency variance
        inst_freq = np.diff(np.unwrap(phase)) * self.sample_rate / (2 * np.pi)
        inst_freq_var = float(np.var(inst_freq)) if len(inst_freq) > 0 else 0.0

        # Spectral centroid
        if power_lin.sum() > 0 and len(freqs) > 0:
            spec_centroid = float(np.sum(freqs * power_lin) / power_lin.sum())
        else:
            spec_centroid = 0.0

        # Spectral rolloff (90%)
        cumulative = np.cumsum(power_lin)
        if cumulative[-1] > 0 and len(freqs) > 0:
            rolloff_idx = np.searchsorted(cumulative, 0.9 * cumulative[-1])
            spec_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        else:
            spec_rolloff = 0.0

        # Presence in water hole band
        water_hole_power = float(np.mean(power_lin[
            (freqs >= 0) & (freqs <= (WATER_HOLE_HIGH_MHZ - WATER_HOLE_LOW_MHZ) * 1e6)
        ])) if len(freqs) > 0 else 0.0

        return {
            "peak_freq_hz":       peak_freq,
            "bw_10db_hz":         bw_10db,
            "spec_entropy":       spec_entropy,
            "spec_flatness":      spec_flatness,
            "mean_envelope":      mean_env,
            "std_envelope":       std_env,
            "kurtosis_envelope":  kurt_env,
            "skew_envelope":      skew_env,
            "mean_phase_diff":    mean_phase_diff,
            "std_phase_diff":     std_phase_diff,
            "second_peak_ac":     second_peak_ac,
            "periodicity_score":  periodicity_score,
            "drift_rate_hz_s":    drift_rate,
            "crest_factor":       crest_factor,
            "am_index":           am_index,
            "fm_index":           fm_index,
            "zcr":                zcr,
            "inst_freq_var":      inst_freq_var,
            "spec_centroid":      spec_centroid,
            "spec_rolloff":       spec_rolloff,
            "water_hole_power":   water_hole_power,
            "signal_power":       float(np.mean(env ** 2)),
            "peak_power":         float(np.max(env ** 2)),
            "log_peak_power":     float(np.log10(np.max(env ** 2) + 1e-20)),
            "snr_estimate":       float(np.max(pdb) - np.median(pdb)) if len(pdb) > 0 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CLASSIFIER — Ensemble ML
# ─────────────────────────────────────────────────────────────────────────────

class SignalClassifier:
    """
    Ensemble classifier: Random Forest + Gradient Boosting + SVM.
    Trained on synthetic feature vectors with data augmentation.
    """

    CLASSES = [
        SignalClass.NARROWBAND_CW,
        SignalClass.NARROWBAND_PULSED,
        SignalClass.PULSAR,
        SignalClass.CHIRP,
        SignalClass.BROADBAND_BURST,
        SignalClass.STRUCTURED_BPSK,
        SignalClass.STRUCTURED_FSK,
        SignalClass.ASTROPHYSICAL_LINE,
        SignalClass.ANOMALOUS,
    ]

    def __init__(self):
        self.rf  = RandomForestClassifier(n_estimators=200, max_depth=12,
                                          min_samples_leaf=2, n_jobs=-1,
                                          random_state=42, class_weight="balanced")
        self.gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08,
                                              max_depth=5, random_state=42)
        self.svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                       random_state=42, class_weight="balanced")
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.le.fit([c.value for c in self.CLASSES])
        self.trained = False
        self._feature_importances: Optional[np.ndarray] = None

    def _synthetic_features(self, signal_class: SignalClass, n_samples: int,
                             rng: np.random.Generator) -> np.ndarray:
        """Generate synthetic feature distributions per class for training."""
        # Each class has distinctive feature signatures
        templates = {
            SignalClass.NARROWBAND_CW: dict(
                spec_entropy=(1.5, 0.3), spec_flatness=(0.01, 0.005),
                bw_10db_hz=(200, 100), am_index=(0.05, 0.02),
                fm_index=(0.1, 0.05), periodicity_score=(0.1, 0.05),
                drift_rate_hz_s=(0.1, 0.5), crest_factor=(1.5, 0.3),
                kurtosis_envelope=(3.0, 0.5), snr_estimate=(20, 5)),
            SignalClass.NARROWBAND_PULSED: dict(
                spec_entropy=(2.0, 0.4), spec_flatness=(0.02, 0.01),
                bw_10db_hz=(300, 100), am_index=(0.8, 0.2),
                fm_index=(0.1, 0.05), periodicity_score=(0.7, 0.15),
                drift_rate_hz_s=(0.05, 0.2), crest_factor=(4.0, 1.0),
                kurtosis_envelope=(8.0, 2.0), snr_estimate=(15, 5)),
            SignalClass.PULSAR: dict(
                spec_entropy=(2.5, 0.4), spec_flatness=(0.05, 0.02),
                bw_10db_hz=(50000, 20000), am_index=(0.9, 0.2),
                fm_index=(0.05, 0.02), periodicity_score=(0.85, 0.1),
                drift_rate_hz_s=(0.01, 0.05), crest_factor=(10, 3),
                kurtosis_envelope=(15, 5), snr_estimate=(10, 3)),
            SignalClass.CHIRP: dict(
                spec_entropy=(4.0, 0.5), spec_flatness=(0.3, 0.1),
                bw_10db_hz=(500000, 100000), am_index=(0.3, 0.1),
                fm_index=(5.0, 2.0), periodicity_score=(0.05, 0.05),
                drift_rate_hz_s=(500, 200), crest_factor=(2.0, 0.5),
                kurtosis_envelope=(2.5, 0.5), snr_estimate=(12, 4)),
            SignalClass.BROADBAND_BURST: dict(
                spec_entropy=(5.5, 0.5), spec_flatness=(0.5, 0.15),
                bw_10db_hz=(1e6, 3e5), am_index=(0.7, 0.2),
                fm_index=(1.0, 0.4), periodicity_score=(0.05, 0.05),
                drift_rate_hz_s=(1.0, 5.0), crest_factor=(3.0, 1.0),
                kurtosis_envelope=(5.0, 2.0), snr_estimate=(8, 3)),
            SignalClass.STRUCTURED_BPSK: dict(
                spec_entropy=(3.0, 0.3), spec_flatness=(0.15, 0.05),
                bw_10db_hz=(10000, 3000), am_index=(0.02, 0.01),
                fm_index=(0.3, 0.1), periodicity_score=(0.5, 0.1),
                drift_rate_hz_s=(0.2, 0.5), crest_factor=(1.4, 0.2),
                kurtosis_envelope=(1.5, 0.3), snr_estimate=(18, 4)),
            SignalClass.STRUCTURED_FSK: dict(
                spec_entropy=(2.8, 0.3), spec_flatness=(0.08, 0.03),
                bw_10db_hz=(15000, 5000), am_index=(0.03, 0.01),
                fm_index=(2.0, 0.5), periodicity_score=(0.4, 0.1),
                drift_rate_hz_s=(0.2, 0.5), crest_factor=(1.5, 0.2),
                kurtosis_envelope=(1.7, 0.3), snr_estimate=(15, 3)),
            SignalClass.ASTROPHYSICAL_LINE: dict(
                spec_entropy=(1.8, 0.4), spec_flatness=(0.005, 0.003),
                bw_10db_hz=(150, 80), am_index=(0.02, 0.01),
                fm_index=(0.05, 0.02), periodicity_score=(0.02, 0.02),
                drift_rate_hz_s=(0.08, 0.3), crest_factor=(2.0, 0.5),
                kurtosis_envelope=(2.8, 0.5), snr_estimate=(8, 3)),
            SignalClass.ANOMALOUS: dict(
                spec_entropy=(4.5, 1.0), spec_flatness=(0.2, 0.15),
                bw_10db_hz=(50000, 30000), am_index=(0.5, 0.3),
                fm_index=(3.0, 2.0), periodicity_score=(0.3, 0.2),
                drift_rate_hz_s=(5.0, 10.0), crest_factor=(5.0, 3.0),
                kurtosis_envelope=(6.0, 4.0), snr_estimate=(5, 3)),
        }
        tmpl = templates[signal_class]
        feature_keys = list(DSPEngine().extract_features(
            np.zeros(1024, dtype=complex)).keys())
        n_features = len(feature_keys)
        X = rng.normal(0, 0.1, (n_samples, n_features))
        key_map = {
            "spec_entropy": 2, "spec_flatness": 3, "bw_10db_hz": 1,
            "am_index": 14, "fm_index": 15, "periodicity_score": 11,
            "drift_rate_hz_s": 12, "crest_factor": 13,
            "kurtosis_envelope": 6, "snr_estimate": 24,
        }
        for feat, idx in key_map.items():
            if feat in tmpl and idx < n_features:
                mu, sigma = tmpl[feat]
                X[:, idx] = rng.normal(mu, sigma, n_samples)
        return X

    def train(self, n_per_class: int = 300) -> Dict[str, Any]:
        """Train ensemble on synthetic data with augmentation."""
        rng = np.random.default_rng(42)
        all_X, all_y = [], []
        for cls in self.CLASSES:
            X = self._synthetic_features(cls, n_per_class, rng)
            y = [cls.value] * n_per_class
            all_X.append(X)
            all_y.extend(y)
        X_train = np.vstack(all_X)
        y_train = np.array(all_y)
        y_enc = self.le.transform(y_train)
        X_scaled = self.scaler.fit_transform(X_train)
        self.rf.fit(X_scaled, y_enc)
        self.gb.fit(X_scaled, y_enc)
        self.svm.fit(X_scaled, y_enc)
        self.trained = True
        self._feature_importances = self.rf.feature_importances_
        # Cross-val on RF
        cv_scores = cross_val_score(self.rf, X_scaled, y_enc, cv=5, scoring="f1_weighted")
        return {
            "cv_f1_mean": float(np.mean(cv_scores)),
            "cv_f1_std":  float(np.std(cv_scores)),
            "n_samples":  len(y_train),
            "n_classes":  len(self.CLASSES),
        }

    def predict(self, features: Dict[str, float]
                ) -> Tuple[SignalClass, float, Dict[str, float]]:
        """
        Ensemble predict. Returns (class, confidence, class_probabilities).
        """
        if not self.trained:
            self.train()
        x = np.array(list(features.values())).reshape(1, -1)
        # Pad/trim to expected feature size
        n_expected = self.rf.n_features_in_
        if x.shape[1] < n_expected:
            x = np.hstack([x, np.zeros((1, n_expected - x.shape[1]))])
        elif x.shape[1] > n_expected:
            x = x[:, :n_expected]
        x_scaled = self.scaler.transform(x)
        prob_rf  = self.rf.predict_proba(x_scaled)[0]
        prob_gb  = self.gb.predict_proba(x_scaled)[0]
        prob_svm = self.svm.predict_proba(x_scaled)[0]
        # Weighted ensemble
        prob_ensemble = 0.5 * prob_rf + 0.3 * prob_gb + 0.2 * prob_svm
        pred_idx = int(np.argmax(prob_ensemble))
        confidence = float(prob_ensemble[pred_idx])
        pred_class_value = self.le.inverse_transform([pred_idx])[0]
        pred_class = SignalClass(pred_class_value)
        class_probs = {
            self.le.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(prob_ensemble)
        }
        return pred_class, confidence, class_probs


# ─────────────────────────────────────────────────────────────────────────────
# WOW FACTOR SCORING — SETI significance metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_wow_factor(record: SignalRecord, features: Dict[str, float]) -> float:
    """
    Compute SETI significance score (0–10 scale).
    Modeled after the original Wow! signal analysis criteria.
    """
    score = 0.0

    # 1. SNR contribution (max 2.5 pts)
    snr_score = min(2.5, max(0.0, record.snr_db / 30.0 * 2.5))
    score += snr_score

    # 2. Narrowness (max 1.5 pts) — narrowband = more interesting
    bw = features.get("bw_10db_hz", 1e6)
    bw_score = 1.5 * max(0.0, 1.0 - min(1.0, bw / 10000.0))
    score += bw_score

    # 3. Water hole proximity (max 1.5 pts)
    freq_mhz = record.center_freq_mhz
    h_dist = abs(freq_mhz - HYDROGEN_LINE_MHZ)
    oh_dist = abs(freq_mhz - 1666.0)
    wh_dist = min(h_dist, oh_dist)
    wh_score = 1.5 * max(0.0, 1.0 - wh_dist / 50.0)
    score += wh_score

    # 4. Drift rate (max 1.0 pt) — stellar-rate drift is astrophysically consistent
    drift = abs(features.get("drift_rate_hz_s", 0.0))
    # Expected Earth drift rate ~0.01–0.3 Hz/s; > 1 Hz/s is suspicious
    if 0.001 < drift < 0.4:
        drift_score = 1.0
    elif drift < 2.0:
        drift_score = 0.5
    else:
        drift_score = 0.1  # Too fast — likely RFI or anomalous
    score += drift_score

    # 5. Spectral purity (max 1.0 pt)
    flatness = features.get("spec_flatness", 0.5)
    purity_score = 1.0 * max(0.0, 1.0 - flatness)
    score += purity_score

    # 6. Anomaly bonus (max 1.5 pts)
    if record.signal_class in (SignalClass.ANOMALOUS, SignalClass.ARIRAL, SignalClass.VOID_CARRIER):
        score += 1.5
    elif record.signal_class in (SignalClass.STRUCTURED_BPSK, SignalClass.STRUCTURED_FSK):
        score += 0.8

    # 7. High DM = extragalactic or no RFI origin (max 0.5 pts)
    dm_score = min(0.5, record.dispersion_measure / 100.0 * 0.5)
    score += dm_score

    return min(10.0, round(score, 3))


# ─────────────────────────────────────────────────────────────────────────────
# DRIVE MANAGEMENT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Drive:
    drive_id:    str          = field(default_factory=lambda: f"DRV-{uuid.uuid4().hex[:6].upper()}")
    capacity_mb: float        = 512.0
    used_mb:     float        = 0.0
    records:     List[str]    = field(default_factory=list)  # List of UIDs
    sealed:      bool         = False
    corrupted:   bool         = False
    hq_hash:     Optional[str] = None

    @property
    def free_mb(self) -> float:
        return self.capacity_mb - self.used_mb

    @property
    def fill_fraction(self) -> float:
        return self.used_mb / self.capacity_mb

    def write_record(self, record: SignalRecord, size_mb: float = 5.0) -> bool:
        if self.sealed or self.corrupted:
            return False
        if self.used_mb + size_mb > self.capacity_mb:
            return False
        self.records.append(record.uid)
        self.used_mb += size_mb
        return True

    def seal(self) -> str:
        """Seal drive and compute HQ hash code."""
        self.sealed = True
        content = "|".join(sorted(self.records))
        self.hq_hash = hashlib.sha256(content.encode()).hexdigest()[:20].upper()
        return self.hq_hash


class DriveManager:
    """Manages physical drive inventory — direct VotV mechanic."""

    def __init__(self):
        self.drives: Dict[str, Drive] = {}
        self.active_drive_id: Optional[str] = None
        self.erased_uids: List[str] = []

    def insert_drive(self, capacity_mb: float = 512.0) -> Drive:
        d = Drive(capacity_mb=capacity_mb)
        self.drives[d.drive_id] = d
        if self.active_drive_id is None:
            self.active_drive_id = d.drive_id
        return d

    @property
    def active_drive(self) -> Optional[Drive]:
        if self.active_drive_id and self.active_drive_id in self.drives:
            return self.drives[self.active_drive_id]
        return None

    def write(self, record: SignalRecord) -> Tuple[bool, str]:
        d = self.active_drive
        if d is None:
            return False, "NO_DRIVE_INSERTED"
        # VOID_CARRIER requires immediate drive erasure protocol
        if record.signal_class == SignalClass.VOID_CARRIER:
            return False, "VOID_CARRIER_WRITE_BLOCKED — ERASE DRIVE NOW"
        ok = d.write_record(record)
        if not ok:
            return False, "DRIVE_FULL_OR_SEALED"
        return True, d.drive_id

    def erase_drive(self, drive_id: str) -> bool:
        if drive_id in self.drives:
            d = self.drives[drive_id]
            self.erased_uids.extend(d.records)
            del self.drives[drive_id]
            if self.active_drive_id == drive_id:
                self.active_drive_id = next(iter(self.drives), None)
            return True
        return False

    def seal_and_submit(self, drive_id: str) -> Optional[str]:
        if drive_id in self.drives:
            return self.drives[drive_id].seal()
        return None

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for did, d in self.drives.items():
            rows.append({
                "Drive ID": did,
                "Used (MB)": round(d.used_mb, 1),
                "Capacity (MB)": d.capacity_mb,
                "Fill %": round(d.fill_fraction * 100, 1),
                "Records": len(d.records),
                "Sealed": d.sealed,
                "Corrupted": d.corrupted,
                "HQ Hash": d.hq_hash or "—",
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    if "drive_manager" not in st.session_state:
        dm = DriveManager()
        dm.insert_drive(512.0)
        st.session_state.drive_manager = dm
    if "classifier" not in st.session_state:
        clf = SignalClassifier()
        clf.train()
        st.session_state.classifier = clf
    if "dsp" not in st.session_state:
        st.session_state.dsp = DSPEngine(sample_rate=2e6)
    if "generator" not in st.session_state:
        st.session_state.generator = SignalGenerator(sample_rate=2e6)
    if "total_points" not in st.session_state:
        st.session_state.total_points = 0
    if "day" not in st.session_state:
        st.session_state.day = 1
    if "threat_level" not in st.session_state:
        st.session_state.threat_level = ThreatLevel.NOMINAL


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def signal_engine_page():
    init_session_state()

    st.markdown("""
    <style>
    .eng-header {
        font-family: 'Courier New', monospace;
        color: #00ff88;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #00ff8840;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .eng-label {
        font-family: 'Courier New', monospace;
        color: #88ffcc;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
    }
    .metric-box {
        background: #050f08;
        border: 1px solid #00ff4433;
        border-radius: 2px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .alert-void {
        background: #1a0000;
        border: 2px solid #ff0000;
        color: #ff4444;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        padding: 0.6rem;
        letter-spacing: 0.1em;
        animation: blink 1s linear infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eng-header">[ SIGNAL ACQUISITION & PROCESSING CONSOLE — ASO-DUNKELTALER ]</div>', unsafe_allow_html=True)

    clf: SignalClassifier = st.session_state.classifier
    dsp: DSPEngine = st.session_state.dsp
    gen: SignalGenerator = st.session_state.generator
    dm: DriveManager = st.session_state.drive_manager

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="eng-label">— SIGNAL PARAMETERS —</div>', unsafe_allow_html=True)

        sig_class_name = st.selectbox(
            "Signal Class Override",
            [c.value for c in SignalClass if c != SignalClass.VOID_CARRIER],
            index=0
        )
        center_freq = st.number_input("Center Frequency (MHz)", 100.0, 10000.0,
                                       float(HYDROGEN_LINE_MHZ), step=0.001, format="%.6f")
        duration    = st.slider("Duration (s)", 0.5, 20.0, 5.0, 0.5)
        snr_db      = st.slider("Target SNR (dB)", -10.0, 40.0, 15.0, 0.5)
        dm_val      = st.number_input("Dispersion Measure (pc·cm⁻³)", 0.0, 2000.0, 0.0, step=1.0)
        ra_h        = st.number_input("RA (hours)", 0.0, 24.0, 12.345, step=0.001)
        dec_d       = st.number_input("Dec (degrees)", -90.0, 90.0, -29.0, step=0.001)

        st.markdown('<div class="eng-label">— FILTER CONFIG —</div>', unsafe_allow_html=True)
        filter_type = st.selectbox("Filter Type", ["butter", "cheby1", "ellip"])
        filter_bw   = st.number_input("Filter Bandwidth (Hz)", 100.0, 1e6, 50000.0, step=100.0)

        st.markdown('<div class="eng-label">— DRIVE STATUS —</div>', unsafe_allow_html=True)
        active = dm.active_drive
        if active:
            fill_pct = active.fill_fraction * 100
            st.progress(active.fill_fraction, text=f"{fill_pct:.1f}% USED")
            st.markdown(f'<div class="eng-label">ID: {active.drive_id} | '
                        f'{active.free_mb:.0f} MB FREE | '
                        f'{len(active.records)} RECORDS</div>', unsafe_allow_html=True)
        else:
            st.error("NO DRIVE INSERTED")

        if st.button("⬡ INSERT NEW DRIVE"):
            d = dm.insert_drive(512.0)
            st.success(f"Drive inserted: {d.drive_id}")

        if active and not active.sealed:
            if st.button("⬡ SEAL & SUBMIT TO HQ"):
                hq_hash = dm.seal_and_submit(active.drive_id)
                pts = len(active.records) * 50
                st.session_state.total_points += pts
                st.success(f"HASH: {hq_hash} | +{pts} pts")

    with col_right:
        if st.button("▶ ACQUIRE & PROCESS SIGNAL", use_container_width=True):
            sig_class = SignalClass(sig_class_name)
            with st.spinner("[ ACQUIRING ... ]"):
                # Generate signal
                t, raw_signal = _generate_for_class(gen, sig_class, duration, snr_db, dm_val)

                # Filter
                half_bw = filter_bw / 2
                filtered = dsp.apply_bandpass(raw_signal, 1.0, half_bw, filter_type=filter_type)

                # Extract features
                features = dsp.extract_features(filtered, freq_mhz=center_freq)
                # Detect drift
                drift, _ = dsp.detect_drift_rate(filtered)
                features["drift_rate_hz_s"] = drift

                # Classify
                pred_class, confidence, class_probs = clf.predict(features)

                # Build record
                period = dsp.detect_periodicity(filtered) if sig_class in (
                    SignalClass.PULSAR, SignalClass.NARROWBAND_PULSED) else None
                snr_measured = dsp.estimate_snr(
                    filtered,
                    (-half_bw * 0.1, half_bw * 0.1),
                    [(-half_bw, -half_bw * 0.2), (half_bw * 0.2, half_bw)]
                )

                record = SignalRecord(
                    signal_class=pred_class if confidence > 0.5 else sig_class,
                    origin=_infer_origin(dm_val, center_freq),
                    center_freq_mhz=center_freq,
                    bandwidth_hz=features.get("bw_10db_hz", filter_bw),
                    drift_rate_hz_s=drift,
                    duration_s=duration,
                    sample_rate_hz=gen.sample_rate,
                    period_s=period,
                    snr_db=snr_measured,
                    dispersion_measure=dm_val,
                    ra_hours=ra_h,
                    dec_degrees=dec_d,
                    classifier_confidence=confidence,
                    stage=ProcessingStage.CLASSIFIED,
                )
                record.wow_factor = compute_wow_factor(record, features)
                record.threat_level = _assess_threat(record)

                # Drive write
                write_ok, write_msg = dm.write(record)
                record.stage = ProcessingStage.ARCHIVED if write_ok else ProcessingStage.CLASSIFIED

                st.session_state.signal_log.append(record.to_dataframe_row())

                # Points
                if write_ok:
                    pts = int(record.wow_factor * 100 + record.snr_db * 5)
                    st.session_state.total_points += pts

            # VOID CARRIER alert
            if record.signal_class == SignalClass.VOID_CARRIER:
                st.markdown('<div class="alert-void">⚠ VOID_CARRIER DETECTED — ERASE DRIVE — DO NOT TRANSMIT ⚠</div>',
                            unsafe_allow_html=True)

            # Display results
            st.markdown('<div class="eng-label">— ACQUISITION RESULTS —</div>', unsafe_allow_html=True)
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Classification", record.signal_class.value.replace("_", " "))
            rc2.metric("Confidence", f"{confidence * 100:.1f}%")
            rc3.metric("SNR", f"{record.snr_db:.1f} dB")
            rc4.metric("WoW Factor", f"{record.wow_factor:.2f}/10")

            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("Drift Rate", f"{drift:.3f} Hz/s")
            rr2.metric("Threat", record.threat_level.name)
            rr3.metric("Drive Write", "✓ OK" if write_ok else f"✗ {write_msg[:12]}")

            # Spectrum plot
            freqs, pdb = dsp.compute_fft(filtered, window="blackman_harris")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), facecolor="#020b05")
            # Power spectrum
            ax1.plot(freqs / 1e3, pdb, color="#00ff88", lw=0.8, alpha=0.9)
            ax1.fill_between(freqs / 1e3, pdb, pdb.min(), alpha=0.12, color="#00ff88")
            ax1.set_facecolor("#020b05")
            ax1.set_xlabel("Freq offset (kHz)", color="#88ffcc", fontsize=7, fontfamily="monospace")
            ax1.set_ylabel("Power (dB)", color="#88ffcc", fontsize=7, fontfamily="monospace")
            ax1.tick_params(colors="#44aa66", labelsize=6)
            for spine in ax1.spines.values():
                spine.set_edgecolor("#00ff4433")
            ax1.set_title("POWER SPECTRUM", color="#00ff88", fontsize=8,
                          fontfamily="monospace", loc="left")

            # Waterfall
            times, wf_freqs, stft_db = dsp.compute_stft(
                filtered, n_fft=256, hop=64, window="hann")
            im = ax2.imshow(stft_db, aspect="auto", origin="lower",
                            extent=[times[0], times[-1],
                                    wf_freqs[0] / 1e3, wf_freqs[-1] / 1e3],
                            cmap="inferno", vmin=np.percentile(stft_db, 5),
                            vmax=np.percentile(stft_db, 99))
            ax2.set_facecolor("#020b05")
            ax2.set_xlabel("Time (s)", color="#88ffcc", fontsize=7, fontfamily="monospace")
            ax2.set_ylabel("Freq (kHz)", color="#88ffcc", fontsize=7, fontfamily="monospace")
            ax2.tick_params(colors="#44aa66", labelsize=6)
            ax2.set_title("WATERFALL", color="#00ff88", fontsize=8,
                          fontfamily="monospace", loc="left")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#00ff4433")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Class probabilities
            st.markdown('<div class="eng-label">— CLASS PROBABILITY DISTRIBUTION —</div>', unsafe_allow_html=True)
            prob_df = pd.DataFrame(
                [(k.replace("_", " "), round(v * 100, 2)) for k, v in
                 sorted(class_probs.items(), key=lambda x: -x[1])],
                columns=["Class", "Probability (%)"]
            )
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Signal log
    st.markdown("---")
    st.markdown('<div class="eng-label">— SIGNAL LOG (SESSION) —</div>', unsafe_allow_html=True)
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.markdown(f'<div class="eng-label">SESSION TOTAL: {st.session_state.total_points} pts | '
                    f'SIGNALS: {len(st.session_state.signal_log)} | '
                    f'DAY: {st.session_state.day}</div>', unsafe_allow_html=True)

        # Drive inventory
        drive_df = dm.summary_df()
        if not drive_df.empty:
            st.markdown('<div class="eng-label">— DRIVE INVENTORY —</div>', unsafe_allow_html=True)
            st.dataframe(drive_df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="eng-label">NO SIGNALS ACQUIRED THIS SESSION. INITIATE ACQUISITION.</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_for_class(gen: SignalGenerator, cls: SignalClass,
                        duration: float, snr_db: float,
                        dm: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(time.time_ns()) % (2**31))
    freq_off = rng.uniform(-1e5, 1e5)
    if cls == SignalClass.NARROWBAND_CW:
        return gen.generate_narrowband_cw(duration, freq_off, snr_db, dm=dm)
    elif cls == SignalClass.NARROWBAND_PULSED:
        period = rng.uniform(0.1, 3.0)
        width = rng.uniform(0.005, period * 0.3)
        return gen.generate_narrowband_pulsed(duration, freq_off, snr_db, period, width, dm=dm)
    elif cls == SignalClass.PULSAR:
        period = rng.choice([0.033, 0.089, 0.714, 1.337, 3.745])
        return gen.generate_pulsar(duration, period, snr_db, max(dm, 10.0),
                                   rng.choice(["gaussian", "double_peaked", "exponential"]))
    elif cls == SignalClass.CHIRP:
        return gen.generate_chirp(duration, snr_db, freq_off, freq_off + rng.uniform(1e4, 5e5), dm=dm)
    elif cls == SignalClass.BROADBAND_BURST:
        return gen.generate_fast_radio_burst(duration, snr_db, max(dm, 50.0),
                                             rng.uniform(1, 30))
    elif cls == SignalClass.STRUCTURED_BPSK:
        bps = rng.choice([100, 1000, 9600, 115200])
        return gen.generate_bpsk(duration, snr_db, bps, freq_off)
    elif cls == SignalClass.STRUCTURED_FSK:
        bps = rng.choice([300, 1200, 9600])
        dev = rng.uniform(500, 5000)
        return gen.generate_fsk(duration, snr_db, bps, freq_off - dev, freq_off + dev)
    elif cls == SignalClass.ASTROPHYSICAL_LINE:
        return gen.generate_narrowband_cw(duration, 0.0, snr_db, bandwidth=50.0, dm=dm)
    elif cls == SignalClass.ANOMALOUS:
        return gen.generate_anomalous(duration, snr_db)
    elif cls == SignalClass.ARIRAL:
        return gen.generate_anomalous(duration, snr_db + 5)
    else:
        return gen.generate_void_carrier(duration)


def _infer_origin(dm: float, freq_mhz: float) -> SignalOrigin:
    if dm > 500:
        return SignalOrigin.EXTRAGALACTIC
    elif dm > 10:
        return SignalOrigin.GALACTIC
    elif dm < 1 and 100 < freq_mhz < 30000:
        return SignalOrigin.SOLAR_SYSTEM
    else:
        return SignalOrigin.UNKNOWN


def _assess_threat(record: SignalRecord) -> ThreatLevel:
    if record.signal_class == SignalClass.VOID_CARRIER:
        return ThreatLevel.CONTAINMENT
    if record.signal_class in (SignalClass.ARIRAL,) and record.wow_factor > 7:
        return ThreatLevel.CRITICAL
    if record.signal_class in (SignalClass.ANOMALOUS, SignalClass.ARIRAL):
        return ThreatLevel.ELEVATED
    if record.signal_class in (SignalClass.STRUCTURED_BPSK, SignalClass.STRUCTURED_FSK) \
            and record.classifier_confidence > 0.8:
        return ThreatLevel.ELEVATED
    return ThreatLevel.NOMINAL


if __name__ == "__main__":
    signal_engine_page()
