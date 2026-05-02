"""
anomaly_detector.py — Anomaly Detection & Threat Assessment Engine
Voices of the Void | Alpen Signal Observatorium — Dunkeltaler Forest
Behavioral Intelligence & Containment Protocol System v4.0.7

Handles: Multi-algorithm anomaly detection (Isolation Forest, LOF, OCSVM,
         CUSUM, Bayesian change-point, HMM), entity classification, sequential
         pattern recognition, ARIRAL reputation mechanics, event taxonomy,
         threat escalation, time-of-day anomaly correlation, multi-variate
         anomaly scoring, real-time monitoring, behavioral fingerprinting,
         containment protocol triggers.

DR. KEL — READ THIS BEFORE TOUCHING ANYTHING IN THIS MODULE.
If `ThreatLevel.CONTAINMENT` is returned, do not continue analysis.
Erase the drive. Close the panel. Do not look outside.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Deque, Dict, Iterator, List,
    Optional, Sequence, Tuple, Union
)

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.stats as sp_stats
from scipy.spatial.distance import mahalanobis
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS — Taxonomies & Status Codes
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyClass(Enum):
    NONE              = "NONE"               # Clean signal — expected statistics
    STATISTICAL       = "STATISTICAL"        # Distributional outlier
    STRUCTURAL        = "STRUCTURAL"         # Non-random internal structure
    TEMPORAL          = "TEMPORAL"           # Time-domain irregularity
    SPECTRAL_GHOST    = "SPECTRAL_GHOST"     # No physical source identified
    REPEATING_PATTERN = "REPEATING_PATTERN"  # Deliberate periodicity
    PRIME_ENCODING    = "PRIME_ENCODING"     # Prime-number-based structure
    FIBONACCI_EMBED   = "FIBONACCI_EMBED"    # Fibonacci sequence present
    VOID_SIGNATURE    = "VOID_SIGNATURE"     # Class VOID — restricted
    ARIRAL_COMMS      = "ARIRAL_COMMS"       # Non-human extraterrestrial
    UNKNOWN           = "UNKNOWN"            # Cannot classify — escalate

class EntityClass(Enum):
    """Mirrors VotV entity taxonomy."""
    NONE              = "NONE"
    ARIRAL            = "ARIRAL"            # Benign extraterrestrial species
    RUFUS             = "RUFUS"             # Hostile entity — do not interact
    INSOMNIAC         = "INSOMNIAC"         # Sleep deprivation manifestation
    GHOST_DEER        = "GHOST_DEER"        # Triggered by 'deer' signal L3
    LOOKER            = "LOOKER"            # The Looker Event — sequence entity
    BAD_SUN           = "BAD_SUN"           # Environmental — evacuate base
    THE_END           = "THE_END"           # VOID_CARRIER class — game over
    UNKNOWN_ENTITY    = "UNKNOWN_ENTITY"    # Unidentified — treat as hostile

class EventType(Enum):
    RANDOM          = "RANDOM"
    SIGNAL_TRIGGERED = "SIGNAL_TRIGGERED"
    TIME_TRIGGERED  = "TIME_TRIGGERED"      # 03:33 events
    STORY_MODE      = "STORY_MODE"
    REPUTATION      = "REPUTATION"
    CONTAINMENT     = "CONTAINMENT"

class ThreatLevel(Enum):
    NOMINAL     = 0
    ELEVATED    = 1
    CRITICAL    = 2
    CONTAINMENT = 3

class RepState(Enum):
    """ARIRAL reputation tiers — -100 to +100."""
    HOSTILE    = "HOSTILE"      # [-100, -50)
    WARY       = "WARY"         # [-50, -10)
    NEUTRAL    = "NEUTRAL"      # [-10, +10)
    FRIENDLY   = "FRIENDLY"     # [+10, +50)
    ALLIED     = "ALLIED"       # [+50, +100]

class ChangepointMethod(Enum):
    CUSUM           = "CUSUM"
    PELT            = "PELT"
    BAYESIAN_ONLINE = "BAYESIAN_ONLINE"
    RUPTURES_BIC    = "RUPTURES_BIC"

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    uid:             str            = field(default_factory=lambda: str(uuid.uuid4())[:12].upper())
    timestamp:       float          = field(default_factory=time.time)
    anomaly_class:   AnomalyClass   = AnomalyClass.NONE
    entity_class:    EntityClass    = EntityClass.NONE
    threat_level:    ThreatLevel    = ThreatLevel.NOMINAL
    event_type:      EventType      = EventType.RANDOM
    confidence:      float          = 0.0       # [0, 1]
    anomaly_score:   float          = 0.0       # normalised [0, 1]
    if_score:        float          = 0.0       # Isolation Forest raw score
    lof_score:       float          = 0.0       # LOF raw score
    ocsvm_score:     float          = 0.0       # One-class SVM raw score
    mahal_distance:  float          = 0.0       # Mahalanobis distance (sigma)
    changepoint_idx: Optional[int]  = None
    prime_test_p:    float          = 1.0       # p-value prime encoding test
    fibonacci_r2:    float          = 0.0       # R² of Fibonacci fit
    pattern_entropy: float          = 0.0       # Kolmogorov complexity proxy
    day_number:      int            = 1
    ingame_time:     float          = 0.0       # 0–24 hours
    notes:           str            = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "UID":            self.uid,
            "Time":           pd.Timestamp(self.timestamp, unit="s").strftime("%H:%M:%S"),
            "Anomaly":        self.anomaly_class.value,
            "Entity":         self.entity_class.value,
            "Threat":         self.threat_level.name,
            "Score":          round(self.anomaly_score, 4),
            "IF":             round(self.if_score, 4),
            "LOF":            round(self.lof_score, 4),
            "OCSVM":          round(self.ocsvm_score, 4),
            "Mahal(σ)":       round(self.mahal_distance, 2),
            "Prime p":        f"{self.prime_test_p:.3e}",
            "Fib R²":         round(self.fibonacci_r2, 4),
            "Entropy":        round(self.pattern_entropy, 4),
            "Confidence":     f"{self.confidence * 100:.1f}%",
            "Notes":          self.notes,
        }


@dataclass
class EventRecord:
    uid:          str         = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    timestamp:    float       = field(default_factory=time.time)
    event_type:   EventType   = EventType.RANDOM
    entity:       EntityClass = EntityClass.NONE
    trigger:      str         = ""
    duration_s:   float       = 0.0
    resolved:     bool        = False
    resolution:   str         = ""
    rep_delta:    int         = 0    # ARIRAL reputation change
    notes:        str         = ""


@dataclass
class RepRecord:
    current:    int   = 0         # -100 to +100
    delta_log:  List[Tuple[float, int, str]] = field(default_factory=list)

    @property
    def state(self) -> RepState:
        if self.current < -50:  return RepState.HOSTILE
        if self.current < -10:  return RepState.WARY
        if self.current < +10:  return RepState.NEUTRAL
        if self.current < +50:  return RepState.FRIENDLY
        return RepState.ALLIED

    def update(self, delta: int, reason: str) -> None:
        self.current = int(np.clip(self.current + delta, -100, 100))
        self.delta_log.append((time.time(), delta, reason))

    def tier_fraction(self) -> float:
        """Normalised progress within current tier [0, 1]."""
        tiers = [(-100, -50), (-50, -10), (-10, 10), (10, 50), (50, 100)]
        for lo, hi in tiers:
            if lo <= self.current < hi:
                return (self.current - lo) / (hi - lo)
        return 1.0


@dataclass
class BehaviouralFingerprint:
    """Characterises a signal source's behavioural signature over time."""
    source_id:          str
    n_observations:     int       = 0
    mean_snr:           float     = 0.0
    snr_variance:       float     = 0.0
    mean_drift_hz_s:    float     = 0.0
    drift_variance:     float     = 0.0
    mean_wow:           float     = 0.0
    periodicity_score:  float     = 0.0     # fraction of obs with period
    anomaly_rate:       float     = 0.0     # fraction flagged as anomalous
    prime_enc_rate:     float     = 0.0
    fib_enc_rate:       float     = 0.0
    last_seen:          float     = field(default_factory=time.time)
    entity_hypothesis:  EntityClass = EntityClass.NONE
    entity_confidence:  float     = 0.0


@dataclass
class Changepoint:
    index:       int
    timestamp_s: float
    confidence:  float
    method:      str
    magnitude:   float      # |mean_after - mean_before| / std_before


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyFeatureExtractor:
    """
    Extracts a 40-dimensional feature vector from a raw complex signal
    specifically optimised for distinguishing structured/artificial anomalies
    from thermal noise and astrophysical sources.
    """

    N_FEATURES = 40

    def __init__(self, sample_rate: float = 2e6):
        self.sr = sample_rate

    def extract(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        if n < 64:
            return np.zeros(self.N_FEATURES)

        env   = np.abs(signal)
        phase = np.angle(signal)
        real  = signal.real
        imag  = signal.imag
        env2  = env ** 2

        # ── Time-domain statistics (10 features) ─────────────────────────
        f1  = float(np.mean(env))
        f2  = float(np.std(env))
        f3  = float(sp_stats.kurtosis(env))
        f4  = float(sp_stats.skew(env))
        f5  = float(np.max(env) / (f1 + 1e-12))              # crest factor
        f6  = float(np.mean(env2))                            # mean power
        f7  = float(np.std(env2))
        f8  = float(np.percentile(env, 99) / (f1 + 1e-12))   # 99th / mean
        f9  = float(np.mean(np.abs(np.diff(env))))            # mean |Δenv|
        f10 = float(np.std(np.diff(env)))

        # ── Phase statistics (6 features) ────────────────────────────────
        unwrapped  = np.unwrap(phase)
        phase_diff = np.diff(unwrapped)
        inst_freq  = phase_diff * self.sr / (2 * np.pi)
        f11 = float(np.mean(np.abs(phase_diff)))
        f12 = float(np.std(phase_diff))
        f13 = float(sp_stats.kurtosis(phase_diff))
        f14 = float(np.std(inst_freq))
        f15 = float(np.max(np.abs(inst_freq)) / (np.mean(np.abs(inst_freq)) + 1e-12))
        f16 = float(sp_stats.skew(inst_freq))

        # ── Spectral features (10 features) ──────────────────────────────
        n_fft = min(1024, n)
        w     = np.hanning(n_fft)
        S     = np.fft.fft(signal[:n_fft] * w, n=n_fft)
        pwr   = np.abs(S[:n_fft // 2]) ** 2 / (n_fft ** 2)
        freqs = np.fft.fftfreq(n_fft, 1.0 / self.sr)[:n_fft // 2]
        pdb   = 10 * np.log10(pwr + 1e-30)
        total = pwr.sum() + 1e-30

        f17 = float(np.sum(freqs * pwr) / total)              # spectral centroid
        f18 = float(np.sqrt(np.sum(((freqs - f17) ** 2) * pwr) / total))  # spec spread
        f19 = float(np.exp(np.mean(np.log(pwr + 1e-30))) / (np.mean(pwr) + 1e-30))  # flatness
        f20 = float(-np.sum((pwr / total) * np.log(pwr / total + 1e-30)))   # spectral entropy
        cumsum_pwr = np.cumsum(pwr)
        rolloff_idx = np.searchsorted(cumsum_pwr, 0.9 * cumsum_pwr[-1])
        f21 = float(freqs[min(rolloff_idx, len(freqs) - 1)])  # rolloff
        f22 = float(pdb.max() - np.median(pdb))               # peak above median (dB)
        f23 = float(np.sum(pdb > (np.median(pdb) + 6))) / len(pdb)  # frac > 6dB above floor
        f24 = float(np.std(pdb))
        peaks, _ = sp_signal.find_peaks(pdb, height=np.median(pdb) + 3, distance=5)
        f25 = float(len(peaks))                                # number of spectral peaks
        f26 = float(np.var(pwr))

        # ── Autocorrelation features (6 features) ─────────────────────────
        ac_input = env2 - env2.mean()
        ac = np.correlate(ac_input, ac_input, mode="full")
        ac = ac[n - 1:] / (ac[n - 1] + 1e-20)
        # First secondary peak location (periodicity indicator)
        ac_trunc = ac[1:min(n // 2, 50000)]
        sec_peaks, sec_props = sp_signal.find_peaks(ac_trunc, height=0.1, distance=10)
        f27 = float(len(sec_peaks))                            # # autocorr peaks
        f28 = float(ac_trunc[sec_peaks[0]]) if len(sec_peaks) > 0 else 0.0  # 1st peak height
        f29 = float(sec_peaks[0] / self.sr) if len(sec_peaks) > 0 else 0.0  # 1st peak lag (s)
        f30 = float(np.mean(ac_trunc[:100] > 0.2))            # frac high AC in early lags
        f31 = float(np.sum(ac_trunc > 0.5))                   # # lags with AC > 0.5
        f32 = float(np.trapz(np.abs(ac_trunc[:min(len(ac_trunc), 1000)])))  # AC energy

        # ── Complexity / structure features (8 features) ──────────────────
        # Lempel-Ziv complexity proxy via run-length encoding
        bits = (env > env.mean()).astype(np.uint8)
        f33 = float(self._run_length_complexity(bits))

        # Higuchi fractal dimension (proxy)
        f34 = float(self._higuchi_fd(env[:min(n, 4096)], k_max=8))

        # Sample entropy (short-term)
        f35 = float(self._sample_entropy(env[:min(n, 2048)], m=2, r_frac=0.2))

        # Ratio of power in prime vs non-prime frequency bins
        prime_mask = np.array([self._is_prime(i) for i in range(1, len(freqs) + 1)])
        prime_power = float(np.mean(pwr[prime_mask[:len(pwr)]]) if prime_mask.any() else 0)
        nonprime_power = float(np.mean(pwr[~prime_mask[:len(pwr)]]) if (~prime_mask[:len(pwr)]).any() else 1e-30)
        f36 = float(prime_power / (nonprime_power + 1e-30))   # prime/nonprime ratio

        # Fibonacci sequence correlation in amplitude envelope
        f37 = float(self._fibonacci_correlation(env[:min(n, 512)]))

        # AM/FM modulation indices
        f38 = float(f2 / (f1 + 1e-12))   # AM index (std/mean of envelope)
        f39 = float(f12 / (f11 + 1e-12)) # FM index (std/mean of |Δφ|)

        # Zero-crossing rate (structural indicator)
        f40 = float(np.mean(np.diff(np.sign(real)) != 0))

        return np.array([
            f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
        ], dtype=np.float64)

    # ── Helper: run-length complexity ────────────────────────────────────────
    @staticmethod
    def _run_length_complexity(seq: np.ndarray) -> float:
        """Normalised Lempel-Ziv-76 complexity."""
        n = len(seq)
        if n == 0:
            return 0.0
        s = "".join(map(str, seq.tolist()))
        c, i, l, k, k_max, n_len = 1, 0, 1, 1, 1, len(s)
        while True:
            if s[i + k - 1] != s[l + k - 1]:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n_len:
                        break
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
            else:
                k += 1
                if l + k > n_len:
                    c += 1
                    break
        b = n / math.log(n + 2, 2)
        return float(c / (b + 1e-10))

    # ── Helper: Higuchi fractal dimension ────────────────────────────────────
    @staticmethod
    def _higuchi_fd(x: np.ndarray, k_max: int = 8) -> float:
        n = len(x)
        lk = np.zeros(k_max)
        for k in range(1, k_max + 1):
            lmk = 0.0
            for m in range(1, k + 1):
                # number of intervals
                n_max = int(math.floor((n - m) / k))
                if n_max < 1:
                    continue
                s = 0.0
                for j in range(1, n_max):
                    s += abs(x[m + j * k - 1] - x[m + (j - 1) * k - 1])
                lmk += s * (n - 1) / (k * n_max * k)
            lk[k - 1] = lmk / k
        valid_k = [k for k in range(1, k_max + 1) if lk[k - 1] > 0]
        if len(valid_k) < 2:
            return 1.0
        log_k = np.log(valid_k)
        log_l = np.log([lk[k - 1] for k in valid_k])
        coeffs = np.polyfit(log_k, log_l, 1)
        return float(coeffs[0])

    # ── Helper: sample entropy ────────────────────────────────────────────────
    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
        n = len(x)
        r = r_frac * float(np.std(x))
        if r == 0 or n < 2 * m + 1:
            return 0.0
        def _count_matches(templates, length):
            count = 0
            for i in range(n - length):
                for j in range(i + 1, n - length):
                    if np.max(np.abs(templates[i] - templates[j])) <= r:
                        count += 1
            return count
        # Template matrix
        tm  = np.array([x[i:i + m]     for i in range(n - m)])
        tm1 = np.array([x[i:i + m + 1] for i in range(n - m - 1)])
        a = _count_matches(tm1, m + 1)
        b = _count_matches(tm, m)
        if b == 0 or a == 0:
            return 0.0
        return float(-math.log(a / b))

    # ── Helper: Fibonacci correlation ─────────────────────────────────────────
    @staticmethod
    def _fibonacci_correlation(x: np.ndarray) -> float:
        """R² of envelope against Fibonacci-index spacing."""
        n = len(x)
        fibs = []
        a, b = 1, 1
        while b < n:
            fibs.append(b)
            a, b = b, a + b
        if len(fibs) < 4:
            return 0.0
        fib_vals = x[[min(f, n - 1) for f in fibs]]
        t = np.arange(len(fib_vals), dtype=float)
        if np.std(fib_vals) == 0 or np.std(t) == 0:
            return 0.0
        r, _ = sp_stats.pearsonr(t, fib_vals)
        return float(r ** 2)

    # ── Helper: prime check ───────────────────────────────────────────────────
    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleAnomalyDetector:
    """
    Ensemble of four unsupervised anomaly detectors:
      1. Isolation Forest (IF)        — efficient for high-dimensional data
      2. Local Outlier Factor (LOF)   — density-based, local neighbourhood
      3. One-Class SVM (OCSVM)        — maximum-margin boundary
      4. Minimum Covariance Det.      — Mahalanobis with robust covariance

    Each detector votes; the ensemble score is a weighted combination
    calibrated to minimise false positives on astrophysical signals.
    """

    # Empirically determined weights: IF is most general; OCSVM most sensitive
    WEIGHTS = {"if": 0.35, "lof": 0.25, "ocsvm": 0.25, "mcd": 0.15}

    # Contamination: expected fraction of anomalies in training data
    CONTAMINATION = 0.05

    def __init__(self, n_features: int = AnomalyFeatureExtractor.N_FEATURES):
        self.n_features = n_features
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=min(20, n_features), whiten=True, random_state=42)

        self.if_model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=self.CONTAMINATION,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
        )
        self.lof_model = LocalOutlierFactor(
            n_neighbors=20,
            algorithm="ball_tree",
            leaf_size=30,
            metric="euclidean",
            contamination=self.CONTAMINATION,
            novelty=True,    # needed for predict on new samples
            n_jobs=-1,
        )
        self.ocsvm_model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=self.CONTAMINATION,
            shrinking=True,
            max_iter=2000,
        )
        self.mcd = MinCovDet(support_fraction=0.85, random_state=42)
        self._mcd_trained = False
        self.trained = False
        self._train_X: Optional[np.ndarray] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def _synthetic_normal_data(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate synthetic 'normal' training data representing
        expected statistical properties of astrophysical signals + RFI.
        """
        X = np.zeros((n, self.n_features))
        # Envelope features — log-normal amplitude typical of noise
        X[:, 0]  = rng.lognormal(0.5, 0.4, n)          # mean env
        X[:, 1]  = rng.lognormal(-0.5, 0.5, n)         # std env
        X[:, 2]  = rng.normal(3.0, 1.5, n)             # kurtosis — near Gaussian
        X[:, 3]  = rng.normal(0.0, 0.5, n)             # skew — near 0
        X[:, 4]  = rng.uniform(1.5, 4.0, n)            # crest factor
        X[:, 5]  = rng.lognormal(0.0, 0.5, n)          # mean power
        X[:, 6]  = rng.lognormal(-0.5, 0.5, n)         # std power
        X[:, 7]  = rng.uniform(1.5, 5.0, n)            # 99th/mean
        X[:, 8]  = rng.lognormal(-2, 0.4, n)           # mean |Δenv|
        X[:, 9]  = rng.lognormal(-3, 0.5, n)           # std Δenv
        # Phase — noisy phase walk
        X[:, 10] = rng.uniform(0.8, 1.2, n)            # mean |Δφ|
        X[:, 11] = rng.uniform(0.1, 0.4, n)            # std Δφ
        X[:, 12] = rng.normal(3.0, 1.0, n)             # kurtosis Δφ
        X[:, 13] = rng.lognormal(3, 1, n)              # std inst_freq
        X[:, 14] = rng.uniform(2, 8, n)                # max/mean inst_freq
        X[:, 15] = rng.normal(0, 0.5, n)               # skew inst_freq
        # Spectral — broadband noise characteristics
        X[:, 16] = rng.uniform(1e5, 9e5, n)            # spec centroid
        X[:, 17] = rng.uniform(1e5, 5e5, n)            # spec spread
        X[:, 18] = rng.uniform(0.3, 0.8, n)            # flatness
        X[:, 19] = rng.uniform(10, 18, n)              # entropy
        X[:, 20] = rng.uniform(5e5, 9e5, n)            # rolloff
        X[:, 21] = rng.uniform(2, 12, n)               # peak above median
        X[:, 22] = rng.uniform(0.0, 0.1, n)            # frac > 6dB
        X[:, 23] = rng.uniform(2, 8, n)                # std pdb
        X[:, 24] = rng.uniform(0, 5, n)                # n_peaks
        X[:, 25] = rng.lognormal(-20, 3, n)            # var pwr
        # Autocorrelation — low for pure noise
        X[:, 26] = rng.uniform(0, 2, n)                # n AC peaks
        X[:, 27] = rng.uniform(0, 0.15, n)             # 1st AC peak height
        X[:, 28] = rng.uniform(0, 0.01, n)             # 1st AC peak lag (s)
        X[:, 29] = rng.uniform(0, 0.05, n)             # frac high AC
        X[:, 30] = rng.uniform(0, 2, n)                # #lags > 0.5
        X[:, 31] = rng.uniform(0, 5, n)                # AC energy
        # Complexity — high for pure noise (random), lower for structured
        X[:, 32] = rng.uniform(0.7, 1.0, n)            # LZ complexity
        X[:, 33] = rng.uniform(1.3, 1.8, n)            # Higuchi FD
        X[:, 34] = rng.uniform(0.5, 2.0, n)            # sample entropy
        X[:, 35] = rng.uniform(0.8, 1.2, n)            # prime ratio
        X[:, 36] = rng.uniform(0.0, 0.1, n)            # Fibonacci R²
        X[:, 37] = rng.uniform(0.0, 0.3, n)            # AM index
        X[:, 38] = rng.uniform(0.8, 1.5, n)            # FM index
        X[:, 39] = rng.uniform(0.3, 0.6, n)            # ZCR
        return X

    def train(self, extra_data: Optional[np.ndarray] = None,
              n_synth: int = 2000) -> Dict[str, Any]:
        """
        Train all four detectors on synthetic + optional real normal data.
        Returns training summary metrics.
        """
        rng = np.random.default_rng(1337)
        X_synth = self._synthetic_normal_data(n_synth, rng)
        if extra_data is not None and len(extra_data) > 0:
            X_train = np.vstack([X_synth, extra_data])
        else:
            X_train = X_synth

        # Remove NaN/inf
        X_train = np.where(np.isfinite(X_train), X_train, 0.0)
        X_scaled = self.scaler.fit_transform(X_train)
        X_pca    = self.pca.fit_transform(X_scaled)
        self._train_X = X_pca

        # Train models
        self.if_model.fit(X_pca)
        self.lof_model.fit(X_pca)
        self.ocsvm_model.fit(X_pca)
        # MCD on PCA-reduced data
        try:
            self.mcd.fit(X_pca)
            self._mcd_trained = True
        except Exception:
            self._mcd_trained = False

        self.trained = True
        return {
            "n_train":      len(X_train),
            "n_synth":      n_synth,
            "pca_variance": float(np.sum(self.pca.explained_variance_ratio_)),
            "pca_components": self.pca.n_components_,
        }

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, features: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Score a single 40-dim feature vector.
        Returns (ensemble_score, if_score, lof_score, ocsvm_score, mahal_sigma).

        Scores are normalised to [0, 1] where 1 = most anomalous.
        """
        if not self.trained:
            self.train()

        x = np.where(np.isfinite(features), features, 0.0).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_pca    = self.pca.transform(x_scaled)

        # IF: raw score is negative log of path length, range approx [-0.5, 0.5]
        if_raw   = float(self.if_model.score_samples(x_pca)[0])
        if_norm  = float(np.clip((-if_raw - 0.0) / 0.6, 0.0, 1.0))

        # LOF: raw score is negative LOF (higher = more anomalous)
        lof_raw  = float(self.lof_model.score_samples(x_pca)[0])
        lof_norm = float(np.clip((-lof_raw - 1.0) / 5.0, 0.0, 1.0))

        # OCSVM: signed distance from hyperplane (positive = inlier)
        ocsvm_raw  = float(self.ocsvm_model.score_samples(x_pca)[0])
        ocsvm_norm = float(np.clip((-ocsvm_raw) / 2.0, 0.0, 1.0))

        # MCD Mahalanobis distance
        if self._mcd_trained:
            try:
                mcd_dist = float(self.mcd.mahalanobis(x_pca)[0])
                # Convert to sigma: chi²(n_components) distribution
                n_c = x_pca.shape[1]
                mahal_sigma = float(np.sqrt(max(0.0, mcd_dist -
                                                chi2_dist.ppf(0.5, df=n_c))))
                mcd_norm = float(np.clip(mahal_sigma / 10.0, 0.0, 1.0))
            except Exception:
                mahal_sigma, mcd_norm = 0.0, 0.0
        else:
            mahal_sigma, mcd_norm = 0.0, 0.0

        # Weighted ensemble
        ensemble = (
            self.WEIGHTS["if"]   * if_norm +
            self.WEIGHTS["lof"]  * lof_norm +
            self.WEIGHTS["ocsvm"] * ocsvm_norm +
            self.WEIGHTS["mcd"]  * mcd_norm
        )
        ensemble = float(np.clip(ensemble, 0.0, 1.0))

        return ensemble, if_norm, lof_norm, ocsvm_norm, mahal_sigma

    def decision(self, ensemble_score: float,
                 thresholds: Tuple[float, float, float] = (0.35, 0.60, 0.82)
                 ) -> ThreatLevel:
        """Map ensemble score to ThreatLevel."""
        lo, mid, hi = thresholds
        if ensemble_score >= hi:    return ThreatLevel.CONTAINMENT
        if ensemble_score >= mid:   return ThreatLevel.CRITICAL
        if ensemble_score >= lo:    return ThreatLevel.ELEVATED
        return ThreatLevel.NOMINAL


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE-POINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class ChangepointDetector:
    """
    Multi-method change-point detection for signal time-series monitoring.
    Detects abrupt distributional shifts that may indicate external events.
    """

    def cusum(self, series: np.ndarray, k: float = 0.5,
              h: float = 5.0) -> List[Changepoint]:
        """
        CUSUM (Cumulative Sum) control chart.
        k = allowable slack (in sigma); h = decision threshold (sigma).
        Detects shifts in mean — bidirectional.
        """
        if len(series) < 10:
            return []
        mu = float(np.mean(series))
        sigma = float(np.std(series))
        if sigma < 1e-10:
            return []
        norm = (series - mu) / sigma
        cp_pos, cp_neg = 0.0, 0.0
        cps: List[Changepoint] = []
        last_cp = 0
        for i, v in enumerate(norm):
            cp_pos = max(0.0, cp_pos + v - k)
            cp_neg = max(0.0, cp_neg - v - k)
            if (cp_pos > h or cp_neg > h) and (i - last_cp) > 5:
                mag = abs(float(np.mean(series[i:min(i + 20, len(series))]) -
                                np.mean(series[max(0, i - 20):i]))) / (sigma + 1e-10)
                cps.append(Changepoint(
                    index=i,
                    timestamp_s=float(i / 100),
                    confidence=float(min(1.0, max(cp_pos, cp_neg) / (h * 2))),
                    method="CUSUM",
                    magnitude=mag,
                ))
                cp_pos, cp_neg = 0.0, 0.0
                last_cp = i
        return cps

    def pelt(self, series: np.ndarray, penalty: float = 10.0,
             min_size: int = 10) -> List[Changepoint]:
        """
        Pruned Exact Linear Time (PELT) algorithm for change-point detection.
        Minimises sum of within-segment costs + penalty * n_changepoints.
        Cost function: negative Gaussian log-likelihood.
        """
        n = len(series)
        if n < 2 * min_size:
            return []

        def _cost(s: int, e: int) -> float:
            seg = series[s:e]
            var = float(np.var(seg))
            if var <= 0:
                return 0.0
            return float((e - s) * (np.log(2 * np.pi * var) + 1))

        # DP + pruning
        F = np.full(n + 1, np.inf)
        F[0] = -penalty
        changepoints_at: List[Optional[int]] = [None] * (n + 1)
        candidates = [0]

        for t in range(min_size, n + 1):
            costs = [F[s] + _cost(s, t) + penalty for s in candidates]
            best_s_idx = int(np.argmin(costs))
            F[t] = costs[best_s_idx]
            changepoints_at[t] = candidates[best_s_idx]
            # Prune: remove candidates where adding them can never improve future
            candidates = [s for s, c in zip(candidates, costs)
                          if c <= F[t]]
            if t + min_size <= n:
                candidates.append(t)

        # Backtrack
        cps_raw: List[int] = []
        t = n
        while changepoints_at[t] is not None and changepoints_at[t] != 0:
            cps_raw.append(changepoints_at[t])
            t = changepoints_at[t]

        sigma_global = float(np.std(series))
        result: List[Changepoint] = []
        for idx in sorted(cps_raw):
            lo = max(0, idx - min_size)
            hi = min(n, idx + min_size)
            mag = abs(float(np.mean(series[idx:hi]) -
                            np.mean(series[lo:idx]))) / (sigma_global + 1e-10)
            result.append(Changepoint(
                index=idx,
                timestamp_s=float(idx / 100),
                confidence=float(np.clip(mag / 5.0, 0.0, 1.0)),
                method="PELT",
                magnitude=mag,
            ))
        return result

    def bayesian_online(self, series: np.ndarray,
                        hazard_rate: float = 0.01,
                        threshold: float = 0.5,
                        ) -> List[Changepoint]:
        """
        Bocd — Bayesian Online Changepoint Detection (Adams & MacKay 2007).
        Conjugate Normal-Gamma model. Returns changepoints where the
        run-length posterior probability of r=0 exceeds threshold.
        """
        n = len(series)
        if n < 10:
            return []

        # Hyperparameters for Normal-Gamma prior
        mu0, kappa0, alpha0, beta0 = float(np.mean(series)), 1.0, 1.0, 1.0
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0
        mu_p    = np.array([mu0])
        kappa_p = np.array([kappa0])
        alpha_p = np.array([alpha0])
        beta_p  = np.array([beta0])
        cps: List[Changepoint] = []
        last_cp = 0

        for t in range(n):
            x = series[t]
            # Predictive probability: Student-t
            nu     = 2 * alpha_p
            var_p  = beta_p * (kappa_p + 1) / (alpha_p * kappa_p + 1e-12)
            with np.errstate(all="ignore"):
                log_pred = sp_stats.t.logpdf(x, df=nu, loc=mu_p,
                                              scale=np.sqrt(var_p + 1e-12))

            # Growth probabilities
            log_R_prev = np.log(R[t, :t + 1] + 1e-300)
            log_growth = log_R_prev + log_pred + np.log(1 - hazard_rate)
            log_cp     = logsumexp(log_R_prev + log_pred) + np.log(hazard_rate)

            # Normalise
            log_R_new = np.concatenate([[log_cp], log_growth])
            log_Z = logsumexp(log_R_new)
            R[t + 1, :t + 2] = np.exp(log_R_new - log_Z)

            # Update sufficient statistics
            kappa_new = np.concatenate([[kappa0], kappa_p + 1])
            mu_new    = np.concatenate([[mu0],
                                        (kappa_p * mu_p + x) / kappa_new[1:]])
            alpha_new = np.concatenate([[alpha0], alpha_p + 0.5])
            beta_new  = np.concatenate([[beta0],
                                        beta_p + 0.5 * kappa_p * (x - mu_p) ** 2 /
                                        (kappa_p + 1)])
            mu_p, kappa_p, alpha_p, beta_p = mu_new, kappa_new, alpha_new, beta_new

            # Changepoint if P(r=0) > threshold
            p_cp = float(R[t + 1, 0])
            if p_cp > threshold and (t - last_cp) > 5:
                sigma_global = float(np.std(series))
                mag = abs(float(np.mean(series[t:min(t + 10, n)]) -
                                np.mean(series[max(0, t - 10):t]))) / (sigma_global + 1e-10)
                cps.append(Changepoint(
                    index=t,
                    timestamp_s=float(t / 100),
                    confidence=float(p_cp),
                    method="BAYESIAN_ONLINE",
                    magnitude=mag,
                ))
                last_cp = t

        return cps

    def detect(self, series: np.ndarray, method: str = "pelt",
               **kwargs) -> List[Changepoint]:
        """Dispatch to named method."""
        m = method.lower()
        if m == "cusum":
            return self.cusum(series, **kwargs)
        elif m == "pelt":
            return self.pelt(series, **kwargs)
        elif m == "bayesian_online":
            return self.bayesian_online(series, **kwargs)
        # Combined: run all and merge
        all_cps = (self.cusum(series) + self.pelt(series) +
                   self.bayesian_online(series))
        # Deduplicate within ±10 samples
        merged: List[Changepoint] = []
        used = set()
        for cp in sorted(all_cps, key=lambda c: -c.confidence):
            if not any(abs(cp.index - m_cp.index) < 10 for m_cp in merged):
                merged.append(cp)
        return sorted(merged, key=lambda c: c.index)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL PATTERN TESTER
# ─────────────────────────────────────────────────────────────────────────────

class StructuralPatternTester:
    """
    Tests whether a signal contains deliberate mathematical structure
    indicative of non-random (potentially artificial) origin.
    """

    # ── Prime encoding test ───────────────────────────────────────────────────

    def prime_frequency_test(self, signal: np.ndarray,
                              n_fft: int = 2048,
                              sample_rate: float = 2e6
                              ) -> Tuple[float, float, List[int]]:
        """
        Test whether spectral peaks cluster at prime-numbered frequency bins.
        H0: peak positions are uniformly distributed.
        Returns (chi2_statistic, p_value, prime_peak_bin_indices).
        """
        pwr = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
        # Find top 30 spectral peaks
        peaks, _ = sp_signal.find_peaks(pwr, height=np.mean(pwr) + 2 * np.std(pwr),
                                         distance=3)
        if len(peaks) < 4:
            return 0.0, 1.0, []

        sieve = self._sieve_primes(n_fft // 2)
        n_total_bins = n_fft // 2
        n_prime_bins = len(sieve)
        frac_prime = n_prime_bins / n_total_bins

        prime_peaks = [p for p in peaks if p in sieve]
        n_pp = len(prime_peaks)
        n_p  = len(peaks)

        # Binomial test: is n_pp significantly greater than expected?
        pval = float(sp_stats.binom_test(n_pp, n_p, frac_prime,
                                          alternative="greater"))
        # Chi2 statistic
        expected = n_p * frac_prime
        chi2 = float((n_pp - expected) ** 2 / (expected + 1e-10))
        return chi2, pval, prime_peaks

    @staticmethod
    def _sieve_primes(limit: int) -> set:
        """Sieve of Eratosthenes."""
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        return {i for i, v in enumerate(sieve) if v}

    # ── Fibonacci embedding test ───────────────────────────────────────────────

    def fibonacci_embedding_test(self, signal: np.ndarray
                                  ) -> Tuple[float, float]:
        """
        Test for Fibonacci sequence structure in the amplitude envelope.
        Uses Kolmogorov-Smirnov test against expected Fibonacci scaling.
        Returns (R2_fit, ks_pvalue).
        """
        env = np.abs(signal)
        n = len(env)
        fibs = []
        a, b = 1, 1
        while b < n:
            fibs.append(b)
            a, b = b, a + b

        if len(fibs) < 6:
            return 0.0, 1.0

        fib_vals = np.array([float(env[min(f, n - 1)]) for f in fibs])
        fib_idx  = np.arange(len(fib_vals), dtype=float)

        # Fit linear model to log of fibonacci-indexed amplitudes
        log_fib_vals = np.log(fib_vals + 1e-10)
        if np.std(log_fib_vals) < 1e-10:
            return 0.0, 1.0

        slope, intercept, r, p, se = sp_stats.linregress(fib_idx, log_fib_vals)
        r2 = float(r ** 2)

        # KS test: are spacings consistent with Fibonacci growth (golden ratio)?
        phi = (1 + math.sqrt(5)) / 2
        expected_ratios = np.full(len(fibs) - 1, phi)
        actual_ratios   = np.array([fibs[i + 1] / fibs[i] for i in range(len(fibs) - 1)],
                                    dtype=float)
        _, ks_p = sp_stats.ks_2samp(actual_ratios, expected_ratios)

        return r2, float(ks_p)

    # ── Kolmogorov complexity proxy ────────────────────────────────────────────

    def kolmogorov_complexity_proxy(self, signal: np.ndarray) -> float:
        """
        Estimate normalised Kolmogorov complexity via compression ratio.
        Uses LZ-based complexity from AnomalyFeatureExtractor for consistency.
        """
        env_bits = (np.abs(signal) > np.median(np.abs(signal))).astype(np.uint8)
        return AnomalyFeatureExtractor._run_length_complexity(env_bits)

    # ── Repetition structure test ──────────────────────────────────────────────

    def repetition_test(self, signal: np.ndarray,
                        sample_rate: float = 2e6,
                        max_period_s: float = 5.0
                        ) -> Tuple[bool, float, float]:
        """
        Detect deliberate repetition beyond what astrophysical sources produce.
        Uses autocorrelation peak strength vs random-signal expectation.
        Returns (is_structured, period_s, confidence).
        """
        env  = np.abs(signal) ** 2
        env -= env.mean()
        n = len(env)
        ac = np.correlate(env, env, mode="full")[n - 1:]
        ac /= (ac[0] + 1e-20)
        max_lag = min(int(max_period_s * sample_rate), n // 2)
        ac_trunc = ac[1:max_lag]

        if len(ac_trunc) < 10:
            return False, 0.0, 0.0

        peaks, props = sp_signal.find_peaks(ac_trunc, height=0.1, prominence=0.05,
                                             distance=max(1, int(0.001 * sample_rate)))
        if len(peaks) == 0:
            return False, 0.0, 0.0

        # Strongest peak
        peak_heights = props["peak_heights"]
        best_idx = int(peaks[np.argmax(peak_heights)])
        best_ac  = float(ac_trunc[best_idx])
        period_s = float((best_idx + 1) / sample_rate)

        # Expected max AC for random signal with n samples
        # E[max AC] ≈ sqrt(2 * log(n)) / sqrt(n) for i.i.d. Gaussian
        expected_max = float(math.sqrt(2 * math.log(max_lag + 1)) / math.sqrt(n))
        sigma_expected = float(1.0 / math.sqrt(n))
        n_sigma = float((best_ac - expected_max) / (sigma_expected + 1e-10))
        confidence = float(min(1.0, n_sigma / 10.0))
        is_structured = bool(n_sigma > 4.0)

        return is_structured, period_s, confidence

    # ── Fractal dimension test ─────────────────────────────────────────────────

    def fractal_test(self, signal: np.ndarray) -> Tuple[float, bool]:
        """
        Compute Higuchi fractal dimension and flag signals with
        anomalously low FD (high regularity = artificial structure).
        Returns (fd, is_anomalously_regular).
        """
        env = np.abs(signal[:min(len(signal), 4096)])
        fd = AnomalyFeatureExtractor._higuchi_fd(env, k_max=10)
        # Pure noise: FD ≈ 1.8–2.0 for complex Gaussian noise
        # Structured signals: FD ≈ 1.0–1.4
        is_regular = bool(fd < 1.5)
        return float(fd), is_regular


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class EntityClassifier:
    """
    Probabilistic entity classification based on anomaly record fingerprints.
    Rules derived from VotV lore and behavioural signatures observed in-game.
    Each entity has a characteristic anomaly profile; we match via scoring.
    """

    ENTITY_PROFILES: Dict[EntityClass, Dict[str, Any]] = {
        EntityClass.ARIRAL: {
            "desc": "Benign extraterrestrial — structured, non-random signals; periodic comms",
            "wow_range":      (5.0, 9.0),
            "anomaly_range":  (0.30, 0.65),
            "prime_test_max": 0.05,           # Low p-value: prime encoding present
            "fib_r2_min":     0.3,
            "periodicity":    True,
            "threat_max":     ThreatLevel.ELEVATED,
            "rep_on_detect":  +5,
            "rep_on_sell":    +10,
        },
        EntityClass.RUFUS: {
            "desc": "Hostile entity — broadband, chaotic, high-energy, non-periodic",
            "wow_range":      (1.0, 4.0),
            "anomaly_range":  (0.70, 1.00),
            "prime_test_max": 1.0,
            "fib_r2_min":     0.0,
            "periodicity":    False,
            "threat_max":     ThreatLevel.CRITICAL,
            "rep_on_detect":  -10,
        },
        EntityClass.INSOMNIAC: {
            "desc": "Manifestation — low-amplitude, high-regularity, near-sleep-cycle freq",
            "wow_range":      (0.5, 3.0),
            "anomaly_range":  (0.40, 0.70),
            "prime_test_max": 0.5,
            "fib_r2_min":     0.0,
            "periodicity":    True,
            "threat_max":     ThreatLevel.ELEVATED,
            "rep_on_detect":  0,
        },
        EntityClass.GHOST_DEER: {
            "desc": "Triggered by 'deer' L3 — decaying amplitude, low frequency",
            "wow_range":      (2.0, 5.0),
            "anomaly_range":  (0.45, 0.75),
            "prime_test_max": 1.0,
            "fib_r2_min":     0.0,
            "periodicity":    False,
            "threat_max":     ThreatLevel.ELEVATED,
            "rep_on_detect":  0,
        },
        EntityClass.LOOKER: {
            "desc": "The Looker Event — sequential signals, prime-encoded, escalating",
            "wow_range":      (7.0, 10.0),
            "anomaly_range":  (0.65, 0.90),
            "prime_test_max": 0.01,
            "fib_r2_min":     0.5,
            "periodicity":    True,
            "threat_max":     ThreatLevel.CRITICAL,
            "rep_on_detect":  0,
        },
        EntityClass.BAD_SUN: {
            "desc": "Environmental — broadband solar flare, evacuate immediately",
            "wow_range":      (0.0, 2.0),
            "anomaly_range":  (0.80, 1.00),
            "prime_test_max": 1.0,
            "fib_r2_min":     0.0,
            "periodicity":    False,
            "threat_max":     ThreatLevel.CONTAINMENT,
            "rep_on_detect":  0,
        },
        EntityClass.THE_END: {
            "desc": "VOID CARRIER CLASS — erase drive, do not transmit, evacuate",
            "wow_range":      (9.0, 10.0),
            "anomaly_range":  (0.90, 1.00),
            "prime_test_max": 1e-6,
            "fib_r2_min":     0.9,
            "periodicity":    True,
            "threat_max":     ThreatLevel.CONTAINMENT,
            "rep_on_detect":  -100,
        },
    }

    def classify(self, record: AnomalyRecord,
                 wow_score: float = 0.0
                 ) -> Tuple[EntityClass, float, Dict[EntityClass, float]]:
        """
        Score record against all entity profiles.
        Returns (best_entity, confidence, all_probs).
        """
        scores: Dict[EntityClass, float] = {}

        for entity, profile in self.ENTITY_PROFILES.items():
            s = 0.0
            # WoW range match
            w_lo, w_hi = profile["wow_range"]
            if w_lo <= wow_score <= w_hi:
                s += 0.25 * (1.0 - abs(wow_score - (w_lo + w_hi) / 2) / ((w_hi - w_lo) / 2 + 1e-6))
            # Anomaly score range
            a_lo, a_hi = profile["anomaly_range"]
            if a_lo <= record.anomaly_score <= a_hi:
                s += 0.25
            # Prime test
            if record.prime_test_p < profile["prime_test_max"] and profile["prime_test_max"] < 1.0:
                s += 0.20
            elif profile["prime_test_max"] >= 1.0 and record.prime_test_p >= 0.05:
                s += 0.10
            # Fibonacci R²
            if record.fibonacci_r2 >= profile["fib_r2_min"]:
                s += 0.15
            # Periodicity
            has_period = record.changepoint_idx is not None or record.fibonacci_r2 > 0.2
            if has_period == profile["periodicity"]:
                s += 0.15
            scores[entity] = float(np.clip(s, 0.0, 1.0))

        # Softmax over scores
        score_vals = np.array(list(scores.values()))
        if score_vals.sum() > 0:
            probs_arr = np.exp(score_vals * 3) / np.exp(score_vals * 3).sum()
        else:
            probs_arr = np.ones(len(scores)) / len(scores)

        probs = {e: float(p) for e, p in zip(scores.keys(), probs_arr)}
        best  = max(probs, key=lambda k: probs[k])
        confidence = float(probs[best])

        # Default NONE if all confidence is low
        if confidence < 0.25 or record.anomaly_score < 0.20:
            best, confidence = EntityClass.NONE, 0.0

        return best, confidence, probs


# ─────────────────────────────────────────────────────────────────────────────
# ARIRAL REPUTATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class AriralReputationManager:
    """
    Full ARIRAL reputation system mirroring VotV's rep mechanic.
    Tracks player actions and computes reputation state transitions.
    Tier boundaries: HOSTILE < -50, WARY < -10, NEUTRAL < 10, FRIENDLY < 50, ALLIED.
    """

    REP_ACTIONS: Dict[str, int] = {
        "ariral_signal_sold":          +10,
        "ariral_signal_detected":       +5,
        "void_carrier_erased":         +15,
        "void_carrier_transmitted":    -100,
        "drive_submitted_clean":        +2,
        "anomalous_signal_sold":        +3,
        "hostile_entity_survived":      +1,
        "satellite_repaired":           +1,
        "satellite_neglected":          -3,
        "signal_corrupted":             -2,
        "sleep_missed":                 -1,
        "rufus_survived":               +5,
        "bad_sun_evacuated":            +8,
        "bad_sun_caught_outside":      -20,
        "looker_event_triggered":       -5,
        "looker_event_completed":      +20,
        "gift_received_from_ariral":    +2,
    }

    TIER_BONUSES: Dict[RepState, Dict[str, float]] = {
        RepState.HOSTILE:  {"signal_value_mult": 0.5, "gift_chance": 0.0, "warn_hostile": True},
        RepState.WARY:     {"signal_value_mult": 0.8, "gift_chance": 0.0, "warn_hostile": False},
        RepState.NEUTRAL:  {"signal_value_mult": 1.0, "gift_chance": 0.02, "warn_hostile": False},
        RepState.FRIENDLY: {"signal_value_mult": 1.3, "gift_chance": 0.08, "warn_hostile": False},
        RepState.ALLIED:   {"signal_value_mult": 1.6, "gift_chance": 0.15, "warn_hostile": False},
    }

    def __init__(self):
        self.rep = RepRecord()
        self._event_log: List[EventRecord] = []

    def apply_action(self, action_key: str, custom_delta: Optional[int] = None) -> int:
        if custom_delta is not None:
            delta = custom_delta
        else:
            delta = self.REP_ACTIONS.get(action_key, 0)
        if delta == 0:
            return 0
        old_state = self.rep.state
        self.rep.update(delta, action_key)
        new_state = self.rep.state
        if old_state != new_state:
            self._event_log.append(EventRecord(
                event_type=EventType.REPUTATION,
                entity=EntityClass.ARIRAL,
                trigger=f"Tier transition {old_state.value} → {new_state.value}",
                notes=f"Action: {action_key}, Delta: {delta:+d}",
            ))
        return delta

    def get_bonuses(self) -> Dict[str, Any]:
        return self.TIER_BONUSES[self.rep.state]

    def gift_check(self) -> bool:
        """Stochastic gift event — ARIRAL leaves shrimp pack after sleep."""
        chance = self.get_bonuses()["gift_chance"]
        return bool(np.random.default_rng().random() < chance)

    def signal_value_multiplier(self) -> float:
        return float(self.get_bonuses()["signal_value_mult"])

    def rep_history_df(self) -> pd.DataFrame:
        if not self.rep.delta_log:
            return pd.DataFrame(columns=["Time", "Delta", "Reason", "Running Total"])
        rows = []
        running = 0
        for ts, delta, reason in self.rep.delta_log:
            running += delta
            rows.append({
                "Time":          pd.Timestamp(ts, unit="s").strftime("%H:%M:%S"),
                "Delta":         f"{delta:+d}",
                "Reason":        reason,
                "Running Total": int(np.clip(running, -100, 100)),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FULL ANOMALY ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyPipeline:
    """
    Orchestrates the full anomaly analysis flow:
    feature extraction → ensemble scoring → change-point detection →
    structural tests → entity classification → threat assessment →
    ARIRAL reputation update → event logging.
    """

    def __init__(self, sample_rate: float = 2e6):
        self.sr = sample_rate
        self.extractor   = AnomalyFeatureExtractor(sample_rate)
        self.detector    = EnsembleAnomalyDetector()
        self.cp_detector = ChangepointDetector()
        self.pattern     = StructuralPatternTester()
        self.entity_clf  = EntityClassifier()
        self._buffer: Deque[np.ndarray] = deque(maxlen=256)
        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=256)
        self._trained = False

    def _ensure_trained(self, extra: Optional[np.ndarray] = None) -> None:
        if not self._trained:
            feat_arr = (np.array(list(self._feature_buffer))
                        if len(self._feature_buffer) > 10 else None)
            self.detector.train(extra_data=feat_arr)
            self._trained = True

    def analyse(self, signal: np.ndarray,
                wow_score: float = 0.0,
                day: int = 1,
                ingame_time: float = 12.0,
                rep_manager: Optional[AriralReputationManager] = None,
                ) -> AnomalyRecord:
        """
        Full pipeline. Returns complete AnomalyRecord.
        """
        self._ensure_trained()

        # ── Feature extraction ────────────────────────────────────────────
        features = self.extractor.extract(signal)
        features = np.where(np.isfinite(features), features, 0.0)
        self._feature_buffer.append(features.copy())

        # ── Ensemble scoring ──────────────────────────────────────────────
        ens, if_s, lof_s, ocsvm_s, mahal = self.detector.score(features)
        threat = self.detector.decision(ens)

        # ── Change-point detection on envelope ────────────────────────────
        env = np.abs(signal)
        # Downsample to manageable length for CUSUM
        env_ds = env[::max(1, len(env) // 2000)]
        cps = self.cp_detector.detect(env_ds, method="cusum")
        cp_idx = cps[0].index if cps else None

        # ── Structural pattern tests ──────────────────────────────────────
        chi2_prime, p_prime, prime_peaks = self.pattern.prime_frequency_test(
            signal, sample_rate=self.sr)
        fib_r2, fib_ks_p = self.pattern.fibonacci_embedding_test(signal)
        is_periodic, period_s, period_conf = self.pattern.repetition_test(
            signal, sample_rate=self.sr)
        fd, is_regular = self.pattern.fractal_test(signal)
        complexity = self.pattern.kolmogorov_complexity_proxy(signal)

        # ── Anomaly class determination ────────────────────────────────────
        anomaly_class = self._determine_anomaly_class(
            ens, p_prime, fib_r2, is_periodic, is_regular, complexity, ingame_time)

        # ── Confidence combination ────────────────────────────────────────
        # High IF + low prime p + high fib R² = high confidence artificial
        confidence = float(np.clip(
            ens * 0.5 +
            (1 - p_prime) * 0.2 +
            fib_r2 * 0.15 +
            period_conf * 0.1 +
            (1 - complexity) * 0.05,
            0.0, 1.0
        ))

        # ── Entity classification ─────────────────────────────────────────
        record = AnomalyRecord(
            anomaly_class=anomaly_class,
            threat_level=threat,
            anomaly_score=ens,
            if_score=if_s,
            lof_score=lof_s,
            ocsvm_score=ocsvm_s,
            mahal_distance=mahal,
            changepoint_idx=cp_idx,
            prime_test_p=p_prime,
            fibonacci_r2=fib_r2,
            pattern_entropy=complexity,
            confidence=confidence,
            day_number=day,
            ingame_time=ingame_time,
        )
        entity, ent_conf, ent_probs = self.entity_clf.classify(record, wow_score)
        record.entity_class = entity
        record.confidence = float((confidence + ent_conf) / 2)

        # ── Build notes ───────────────────────────────────────────────────
        notes_parts = []
        if p_prime < 0.05:
            notes_parts.append(f"PRIME_ENC p={p_prime:.3e}")
        if fib_r2 > 0.3:
            notes_parts.append(f"FIB_EMBED R²={fib_r2:.2f}")
        if is_periodic and period_conf > 0.3:
            notes_parts.append(f"PERIODIC T={period_s:.4f}s ρ={period_conf:.2f}")
        if is_regular:
            notes_parts.append(f"LOW_FD={fd:.2f}")
        if cps:
            notes_parts.append(f"CHANGEPOINT@t={cps[0].index} mag={cps[0].magnitude:.1f}σ")
        if entity != EntityClass.NONE:
            notes_parts.append(f"ENTITY:{entity.value} conf={ent_conf:.2f}")
        record.notes = " | ".join(notes_parts) if notes_parts else "NOMINAL"

        # ── ARIRAL rep update ─────────────────────────────────────────────
        if rep_manager is not None and entity == EntityClass.ARIRAL:
            rep_manager.apply_action("ariral_signal_detected")
        if (rep_manager is not None and
                anomaly_class == AnomalyClass.VOID_SIGNATURE):
            rep_manager.apply_action("void_carrier_erased")

        return record

    def analyse_batch(self, signals: List[np.ndarray],
                      wow_scores: Optional[List[float]] = None,
                      day: int = 1) -> List[AnomalyRecord]:
        """Batch analysis — efficient for session-level monitoring."""
        if wow_scores is None:
            wow_scores = [0.0] * len(signals)
        records = []
        for sig, wow in zip(signals, wow_scores):
            records.append(self.analyse(sig, wow, day=day))
        return records

    @staticmethod
    def _determine_anomaly_class(
        ens: float, p_prime: float, fib_r2: float,
        is_periodic: bool, is_regular: bool,
        complexity: float, ingame_time: float
    ) -> AnomalyClass:
        if ens < 0.15:
            return AnomalyClass.NONE

        # VOID signature: extremely low prime p AND high fib AND low complexity
        if p_prime < 1e-4 and fib_r2 > 0.8 and complexity < 0.3:
            return AnomalyClass.VOID_SIGNATURE

        # ARIRAL comms: prime AND fib AND periodic
        if p_prime < 0.05 and fib_r2 > 0.3 and is_periodic:
            return AnomalyClass.ARIRAL_COMMS

        # Prime encoding alone
        if p_prime < 0.05 and ens > 0.35:
            return AnomalyClass.PRIME_ENCODING

        # Fibonacci embedding alone
        if fib_r2 > 0.5 and ens > 0.3:
            return AnomalyClass.FIBONACCI_EMBED

        # Repeating pattern
        if is_periodic and ens > 0.35:
            return AnomalyClass.REPEATING_PATTERN

        # Structural (regular / low FD)
        if is_regular and ens > 0.4:
            return AnomalyClass.STRUCTURAL

        # 03:33 temporal anomaly window (game-time hours 3.5 ± 0.08)
        if abs(ingame_time - 3.55) < 0.12 and ens > 0.25:
            return AnomalyClass.TEMPORAL

        # Spectral ghost: high ensemble but no structural markers
        if ens > 0.5 and complexity > 0.7:
            return AnomalyClass.SPECTRAL_GHOST

        if ens > 0.35:
            return AnomalyClass.STATISTICAL

        return AnomalyClass.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyVisualizer:
    PAL = {
        "bg":      "#030d07",
        "fg":      "#00ff88",
        "dim":     "#004422",
        "accent":  "#00ffcc",
        "warn":    "#ff8800",
        "danger":  "#ff2222",
        "grid":    "#001a0e",
        "axis":    "#335544",
        "void":    "#8800ff",
    }

    def _base_fig(self, figsize, ncols=1, nrows=1, **gs_kw):
        fig = plt.figure(figsize=figsize, facecolor=self.PAL["bg"])
        gs  = gridspec.GridSpec(nrows, ncols, figure=fig, **gs_kw)
        return fig, gs

    def _style(self, ax, title=""):
        ax.set_facecolor(self.PAL["bg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(self.PAL["dim"])
        ax.tick_params(colors=self.PAL["axis"], labelsize=6)
        ax.grid(True, color=self.PAL["grid"], alpha=0.45, linestyle=":")
        if title:
            ax.set_title(title, color=self.PAL["fg"], fontsize=7.5,
                         loc="left", pad=3, fontfamily="monospace")

    def plot_ensemble_breakdown(self, record: AnomalyRecord,
                                figsize=(9, 3.5)) -> plt.Figure:
        fig, gs = self._base_fig(figsize, ncols=3, nrows=1,
                                  hspace=0.05, wspace=0.4)
        # Gauge plot for ensemble score
        ax_gauge = fig.add_subplot(gs[0])
        theta = np.linspace(0, np.pi, 300)
        r_outer, r_inner = 1.0, 0.55
        zones = [(0.00, 0.35, self.PAL["fg"]),
                 (0.35, 0.60, self.PAL["warn"]),
                 (0.60, 0.82, "#ff4400"),
                 (0.82, 1.00, self.PAL["danger"])]
        for lo, hi, col in zones:
            t_lo = np.pi * (1 - hi)
            t_hi = np.pi * (1 - lo)
            t_arc = np.linspace(t_lo, t_hi, 60)
            x_out = r_outer * np.cos(t_arc)
            y_out = r_outer * np.sin(t_arc)
            x_in  = r_inner * np.cos(t_arc[::-1])
            y_in  = r_inner * np.sin(t_arc[::-1])
            ax_gauge.fill(np.concatenate([x_out, x_in]),
                          np.concatenate([y_out, y_in]),
                          color=col, alpha=0.6)
        # Needle
        score_angle = np.pi * (1 - record.anomaly_score)
        needle_len = 0.85
        ax_gauge.annotate("", xy=(needle_len * np.cos(score_angle),
                                   needle_len * np.sin(score_angle)),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle="-|>", color=self.PAL["accent"],
                                           lw=1.5))
        ax_gauge.set_xlim(-1.1, 1.1)
        ax_gauge.set_ylim(-0.2, 1.1)
        ax_gauge.set_aspect("equal")
        ax_gauge.axis("off")
        ax_gauge.set_facecolor(self.PAL["bg"])
        ax_gauge.text(0, -0.15, f"{record.anomaly_score:.3f}",
                      ha="center", va="top", color=self.PAL["accent"],
                      fontsize=11, fontfamily="monospace", fontweight="bold")
        ax_gauge.set_title("ENSEMBLE SCORE", color=self.PAL["fg"],
                           fontsize=7.5, loc="center", fontfamily="monospace")

        # Detector bar chart
        ax_bar = fig.add_subplot(gs[1])
        labels = ["IF", "LOF", "OCSVM", "Mahal/10"]
        values = [record.if_score, record.lof_score, record.ocsvm_score,
                  min(1.0, record.mahal_distance / 10.0)]
        colors = [self.PAL["fg"] if v < 0.35 else
                  self.PAL["warn"] if v < 0.60 else
                  self.PAL["danger"] for v in values]
        bars = ax_bar.barh(labels, values, color=colors, height=0.5, alpha=0.82)
        ax_bar.set_xlim(0, 1)
        ax_bar.axvline(0.35, color=self.PAL["dim"], lw=0.7, ls=":")
        ax_bar.axvline(0.60, color=self.PAL["warn"], lw=0.7, ls=":", alpha=0.5)
        ax_bar.axvline(0.82, color=self.PAL["danger"], lw=0.7, ls=":", alpha=0.5)
        for bar, val in zip(bars, values):
            ax_bar.text(min(val + 0.01, 0.96), bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=6,
                        color=self.PAL["accent"], fontfamily="monospace")
        self._style(ax_bar, "DETECTOR SCORES")
        ax_bar.set_xlabel("Score (0–1)", fontsize=6, color=self.PAL["fg"])

        # Anomaly class + entity
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis("off")
        ax_info.set_facecolor(self.PAL["bg"])
        lines_txt = [
            ("CLASS",    record.anomaly_class.value),
            ("ENTITY",   record.entity_class.value),
            ("THREAT",   record.threat_level.name),
            ("IF",       f"{record.if_score:.4f}"),
            ("LOF",      f"{record.lof_score:.4f}"),
            ("OCSVM",    f"{record.ocsvm_score:.4f}"),
            ("MAHAL",    f"{record.mahal_distance:.2f}σ"),
            ("PRIME p",  f"{record.prime_test_p:.2e}"),
            ("FIB R²",   f"{record.fibonacci_r2:.3f}"),
            ("ENTROPY",  f"{record.pattern_entropy:.3f}"),
            ("CONF",     f"{record.confidence * 100:.1f}%"),
        ]
        for i, (key, val) in enumerate(lines_txt):
            col = (self.PAL["danger"] if "CONTAINMENT" in val or "VOID" in val else
                   self.PAL["warn"]   if "CRITICAL" in val or "ELEVATED" in val else
                   self.PAL["accent"] if i == 0 else
                   self.PAL["fg"])
            ax_info.text(0.02, 1.0 - i * 0.09, f"{key:<9} {val}",
                         transform=ax_info.transAxes,
                         color=col, fontsize=7, fontfamily="monospace", va="top")
        self._style(ax_info, "RECORD SUMMARY")
        plt.tight_layout(pad=0.5)
        return fig

    def plot_anomaly_timeline(self, records: List[AnomalyRecord],
                               figsize=(10, 4.5)) -> plt.Figure:
        if not records:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=12)
            return fig

        fig, gs = self._base_fig(figsize, nrows=3, ncols=1, hspace=0.05)
        times = np.arange(len(records))
        scores     = [r.anomaly_score for r in records]
        thresholds = [r.threat_level.value for r in records]
        entities   = [r.entity_class.value for r in records]

        # Anomaly score timeline
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(times, scores, color=self.PAL["fg"], lw=0.9, alpha=0.9)
        ax1.fill_between(times, scores, 0, alpha=0.15, color=self.PAL["fg"])
        ax1.axhline(0.35, color=self.PAL["warn"],   lw=0.7, ls="--", alpha=0.7)
        ax1.axhline(0.60, color="#ff4400",           lw=0.7, ls="--", alpha=0.7)
        ax1.axhline(0.82, color=self.PAL["danger"],  lw=0.7, ls="--", alpha=0.7)
        # Changepoints
        for r in records:
            if r.changepoint_idx is not None:
                idx = records.index(r)
                ax1.axvline(idx, color=self.PAL["void"], lw=0.6, alpha=0.5)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(times[0], times[-1])
        self._style(ax1, "ANOMALY SCORE TIMELINE")
        ax1.set_ylabel("Score", fontsize=6, color=self.PAL["fg"])
        ax1.tick_params(labelbottom=False)

        # Threat level heatmap
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        threat_arr = np.array(thresholds).reshape(1, -1)
        ax2.imshow(threat_arr, aspect="auto",
                   extent=[times[0] - 0.5, times[-1] + 0.5, 0, 1],
                   cmap=plt.cm.get_cmap("RdYlGn_r"), vmin=0, vmax=3,
                   interpolation="nearest")
        self._style(ax2, "THREAT LEVEL")
        ax2.set_yticks([])
        ax2.tick_params(labelbottom=False)

        # Entity timeline
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        entity_names = [e.value for e in EntityClass]
        entity_nums  = [entity_names.index(e) for e in entities]
        scatter_cols = [
            self.PAL["danger"] if e in ("THE_END", "RUFUS", "BAD_SUN") else
            self.PAL["warn"]   if e in ("LOOKER", "GHOST_DEER", "INSOMNIAC") else
            self.PAL["accent"] if e == "ARIRAL" else
            self.PAL["dim"]
            for e in entities
        ]
        ax3.scatter(times, entity_nums, c=scatter_cols, s=12, alpha=0.8, zorder=3)
        ax3.set_yticks(range(len(entity_names)))
        ax3.set_yticklabels(entity_names, fontsize=5, color=self.PAL["axis"])
        ax3.set_xlim(times[0], times[-1])
        self._style(ax3, "ENTITY DETECTIONS")
        ax3.set_xlabel("Signal Index", fontsize=6, color=self.PAL["fg"])

        fig.tight_layout(pad=0.4)
        return fig

    def plot_feature_pca(self, records: List[AnomalyRecord],
                          features_list: List[np.ndarray],
                          figsize=(8, 5)) -> plt.Figure:
        if len(records) < 4 or len(features_list) < 4:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "INSUFFICIENT DATA (≥4 SAMPLES REQUIRED)",
                    transform=ax.transAxes, ha="center",
                    color=self.PAL["dim"], fontsize=10, fontfamily="monospace")
            ax.set_facecolor(self.PAL["bg"])
            return fig

        X = np.array([np.where(np.isfinite(f), f, 0) for f in features_list])
        X_scaled = RobustScaler().fit_transform(X)
        n_comp = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_comp < 2:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "PCA REQUIRES ≥2 FEATURES", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=10)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        pca = PCA(n_components=2)
        Z = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        ax.set_facecolor(self.PAL["bg"])

        for r, (z0, z1) in zip(records, Z):
            col = (self.PAL["danger"] if r.threat_level == ThreatLevel.CONTAINMENT else
                   "#ff4400"          if r.threat_level == ThreatLevel.CRITICAL else
                   self.PAL["warn"]   if r.threat_level == ThreatLevel.ELEVATED else
                   self.PAL["fg"])
            size = 20 + r.anomaly_score * 60
            ax.scatter(z0, z1, c=col, s=size, alpha=0.75, edgecolors="none", zorder=3)

        # Convex hull for nominal cluster
        from matplotlib.patches import Ellipse
        nom_idx = [i for i, r in enumerate(records) if r.threat_level == ThreatLevel.NOMINAL]
        if len(nom_idx) > 3:
            nom_Z = Z[nom_idx]
            mu = nom_Z.mean(axis=0)
            std = nom_Z.std(axis=0)
            ell = Ellipse(mu, width=std[0] * 4, height=std[1] * 4, angle=0,
                          fill=False, edgecolor=self.PAL["fg"],
                          linestyle="--", lw=0.8, alpha=0.5)
            ax.add_patch(ell)

        # Legend
        handles = [
            mpatches.Patch(color=self.PAL["fg"],     label="NOMINAL"),
            mpatches.Patch(color=self.PAL["warn"],   label="ELEVATED"),
            mpatches.Patch(color="#ff4400",           label="CRITICAL"),
            mpatches.Patch(color=self.PAL["danger"], label="CONTAINMENT"),
        ]
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  facecolor=self.PAL["bg"], edgecolor=self.PAL["dim"],
                  labelcolor=self.PAL["fg"])
        self._style(ax, f"FEATURE SPACE — PCA ({pca.explained_variance_ratio_[0]*100:.0f}% + "
                         f"{pca.explained_variance_ratio_[1]*100:.0f}%  variance)")
        ax.set_xlabel("PC1", fontsize=6, color=self.PAL["fg"])
        ax.set_ylabel("PC2", fontsize=6, color=self.PAL["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    def plot_reputation_gauge(self, rep: RepRecord, figsize=(6, 3)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        ax.set_facecolor(self.PAL["bg"])

        val = rep.current
        tiers = [
            (-100, -50, self.PAL["danger"],  "HOSTILE"),
            (-50,  -10, self.PAL["warn"],    "WARY"),
            (-10,   10, self.PAL["dim"],     "NEUTRAL"),
            (10,    50, "#00aa55",           "FRIENDLY"),
            (50,   100, self.PAL["fg"],      "ALLIED"),
        ]
        bar_h = 0.35
        for lo, hi, col, label in tiers:
            width = hi - lo
            alpha = 0.8 if lo <= val < hi else 0.25
            rect = plt.Rectangle((lo, 0), width, bar_h, color=col, alpha=alpha)
            ax.add_patch(rect)
            ax.text((lo + hi) / 2, bar_h / 2, label, ha="center", va="center",
                    fontsize=6.5, color="white", fontfamily="monospace",
                    fontweight="bold" if lo <= val < hi else "normal")
        # Current position marker
        ax.axvline(val, color=self.PAL["accent"], lw=2.0, ymin=0, ymax=1)
        ax.scatter([val], [bar_h / 2], color=self.PAL["accent"], s=60, zorder=5)
        ax.text(val, bar_h + 0.05, f"{val:+d}", ha="center", va="bottom",
                color=self.PAL["accent"], fontsize=9, fontfamily="monospace",
                fontweight="bold")

        ax.set_xlim(-105, 105)
        ax.set_ylim(-0.05, 0.7)
        ax.axis("off")
        ax.set_title(f"ARIRAL REPUTATION — {rep.state.value}",
                     color=self.PAL["fg"], fontsize=8, fontfamily="monospace",
                     pad=4, loc="center")
        plt.tight_layout(pad=0.3)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_detector_page():
    from signal_engine import init_session_state, SignalGenerator, SignalClass, _generate_for_class

    init_session_state()

    st.markdown("""
    <style>
    .anom-header {
        font-family:'Courier New',monospace;
        color:#00ff88;
        font-size:0.78rem;
        letter-spacing:0.14em;
        border-bottom:1px solid #00ff4430;
        padding-bottom:0.4rem;
        margin-bottom:1rem;
    }
    .anom-label {
        font-family:'Courier New',monospace;
        color:#88ffcc;
        font-size:0.70rem;
        letter-spacing:0.09em;
        margin-top:0.5rem;
        margin-bottom:0.2rem;
    }
    .threat-nominal    { background:#001a0e; border:1px solid #00ff44; }
    .threat-elevated   { background:#1a0e00; border:1px solid #ff8800; }
    .threat-critical   { background:#1a0200; border:1px solid #ff4400; }
    .threat-containment{ background:#1a0000; border:2px solid #ff0000;
                          animation:flicker 0.8s linear infinite; }
    @keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.35} }
    .threat-box {
        font-family:'Courier New',monospace;
        font-size:0.75rem;
        color:#ff4444;
        padding:0.5rem 0.8rem;
        border-radius:2px;
        margin-bottom:0.6rem;
        letter-spacing:0.08em;
    }
    .rep-pill {
        display:inline-block;
        font-family:'Courier New',monospace;
        font-size:0.68rem;
        padding:0.1rem 0.5rem;
        border-radius:2px;
        margin:0.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="anom-header">[ ANOMALY DETECTION & THREAT ASSESSMENT — BEHAVIORAL INTELLIGENCE MODULE ]</div>',
                unsafe_allow_html=True)

    # ── Session state init ────────────────────────────────────────────────
    if "pipeline" not in st.session_state:
        pipe = AnomalyPipeline(sample_rate=2e6)
        pipe._ensure_trained()
        st.session_state.pipeline = pipe
    if "rep_manager" not in st.session_state:
        st.session_state.rep_manager = AriralReputationManager()
    if "anomaly_records" not in st.session_state:
        st.session_state.anomaly_records = []
    if "anomaly_features" not in st.session_state:
        st.session_state.anomaly_features = []

    pipe: AnomalyPipeline          = st.session_state.pipeline
    rep:  AriralReputationManager  = st.session_state.rep_manager
    gen:  SignalGenerator          = st.session_state.generator
    viz = AnomalyVisualizer()

    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown('<div class="anom-label">— SIGNAL PARAMETERS —</div>', unsafe_allow_html=True)
        sig_type   = st.selectbox("Signal Class", [c.value for c in SignalClass])
        snr_db     = st.slider("SNR (dB)", -5.0, 40.0, 15.0, 1.0)
        duration   = st.slider("Duration (s)", 0.5, 20.0, 5.0, 0.5)
        dm_val     = st.number_input("DM (pc·cm⁻³)", 0.0, 3000.0, 0.0, step=1.0)
        wow_score  = st.slider("WoW Override", 0.0, 10.0, 3.0, 0.1)

        st.markdown('<div class="anom-label">— ANALYSIS CONFIG —</div>', unsafe_allow_html=True)
        ingame_h   = st.slider("In-game Time (h)", 0.0, 23.99, 12.0, 0.01)
        day_num    = st.number_input("Day", 1, 200, int(st.session_state.day), step=1)
        cp_method  = st.selectbox("Changepoint Method", ["cusum", "pelt", "bayesian_online"])
        n_batch    = st.number_input("Batch Signals", 1, 50, 1, step=1)

        st.markdown('<div class="anom-label">— ARIRAL ACTIONS —</div>', unsafe_allow_html=True)
        rep_action = st.selectbox("Apply Reputation Action",
                                  ["— select —"] + list(AriralReputationManager.REP_ACTIONS.keys()))
        if st.button("APPLY ACTION") and rep_action != "— select —":
            delta = rep.apply_action(rep_action)
            st.success(f"Rep: {delta:+d} pts → {rep.rep.current:+d} total ({rep.rep.state.value})")

    with col_main:
        if st.button("▶ RUN ANOMALY ANALYSIS", use_container_width=True):
            sig_class = SignalClass(sig_type)
            records_batch: List[AnomalyRecord] = []
            feats_batch:   List[np.ndarray]    = []

            with st.spinner("[ ANALYSING BEHAVIOURAL SIGNATURES ... ]"):
                for _ in range(int(n_batch)):
                    t, signal = _generate_for_class(gen, sig_class, duration, snr_db, dm_val)
                    features  = pipe.extractor.extract(signal)
                    features  = np.where(np.isfinite(features), features, 0.0)
                    record    = pipe.analyse(
                        signal, wow_score=wow_score,
                        day=int(day_num), ingame_time=float(ingame_h),
                        rep_manager=rep,
                    )
                    # Override changepoint method for last signal
                    env_ds = np.abs(signal)[::max(1, len(signal) // 2000)]
                    cps    = pipe.cp_detector.detect(env_ds, method=cp_method)
                    record.changepoint_idx = cps[0].index if cps else None
                    records_batch.append(record)
                    feats_batch.append(features)

                st.session_state.anomaly_records.extend(records_batch)
                st.session_state.anomaly_features.extend(feats_batch)

            latest = records_batch[-1]

            # ── Threat alert ───────────────────────────────────────────────
            cls_map = {
                ThreatLevel.NOMINAL:     ("threat-nominal",    "[ THREAT LEVEL: NOMINAL — SIGNAL WITHIN EXPECTED PARAMETERS ]"),
                ThreatLevel.ELEVATED:    ("threat-elevated",   "[ THREAT LEVEL: ELEVATED — ANOMALOUS MARKERS DETECTED ]"),
                ThreatLevel.CRITICAL:    ("threat-critical",   "[ THREAT LEVEL: CRITICAL — ENTITY CONFIRMED — INITIATE PROTOCOL ]"),
                ThreatLevel.CONTAINMENT: ("threat-containment","[ ⚠ CONTAINMENT ⚠ — ERASE DRIVE — EVACUATE BASE — DO NOT LOOK OUTSIDE ]"),
            }
            css_cls, msg = cls_map[latest.threat_level]
            st.markdown(f'<div class="threat-box {css_cls}">{msg}</div>',
                        unsafe_allow_html=True)

            # ── Ensemble breakdown ─────────────────────────────────────────
            fig_ens = viz.plot_ensemble_breakdown(latest)
            st.pyplot(fig_ens, use_container_width=True)
            plt.close(fig_ens)

            # ── Metrics row ────────────────────────────────────────────────
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Anomaly Class",  latest.anomaly_class.value.replace("_", " "))
            mc2.metric("Entity",         latest.entity_class.value)
            mc3.metric("Ensemble Score", f"{latest.anomaly_score:.4f}")
            mc4.metric("Prime p-val",    f"{latest.prime_test_p:.3e}")
            mc5.metric("Fib R²",         f"{latest.fibonacci_r2:.3f}")

            # ── Notes ──────────────────────────────────────────────────────
            st.markdown(f'<div class="anom-label">ANALYSIS NOTES: {latest.notes}</div>',
                        unsafe_allow_html=True)

            # ── Batch record table ─────────────────────────────────────────
            if n_batch > 1:
                st.markdown('<div class="anom-label">— BATCH RESULTS —</div>',
                            unsafe_allow_html=True)
                batch_df = pd.DataFrame([r.to_row() for r in records_batch])
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

        # ── Session timeline ───────────────────────────────────────────────
        all_records = st.session_state.anomaly_records
        if all_records:
            st.markdown("---")
            st.markdown('<div class="anom-label">— SESSION ANOMALY TIMELINE —</div>',
                        unsafe_allow_html=True)
            fig_tl = viz.plot_anomaly_timeline(all_records)
            st.pyplot(fig_tl, use_container_width=True)
            plt.close(fig_tl)

            # PCA feature space
            all_feats = st.session_state.anomaly_features
            if len(all_feats) >= 4:
                st.markdown('<div class="anom-label">— FEATURE SPACE PCA —</div>',
                            unsafe_allow_html=True)
                fig_pca = viz.plot_feature_pca(all_records, all_feats)
                st.pyplot(fig_pca, use_container_width=True)
                plt.close(fig_pca)

            # Full record table
            with st.expander("[ SESSION ANOMALY LOG ]"):
                log_df = pd.DataFrame([r.to_row() for r in all_records])
                st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ── ARIRAL Reputation Panel ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="anom-label">— ARIRAL REPUTATION SYSTEM —</div>',
                unsafe_allow_html=True)
    rep_col_gauge, rep_col_info = st.columns([2, 1])
    with rep_col_gauge:
        fig_rep = viz.plot_reputation_gauge(rep.rep)
        st.pyplot(fig_rep, use_container_width=True)
        plt.close(fig_rep)
    with rep_col_info:
        bonuses = rep.get_bonuses()
        st.markdown(f'<div class="anom-label">TIER: {rep.rep.state.value}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="anom-label">VALUE MULT: ×{bonuses["signal_value_mult"]:.1f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="anom-label">GIFT CHANCE: {bonuses["gift_chance"]*100:.0f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="anom-label">HOSTILE WARN: {"YES" if bonuses.get("warn_hostile") else "NO"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="anom-label">LAST DELTA: {rep.rep.delta_log[-1][1]:+d} ({rep.rep.delta_log[-1][2]})</div>'
                    if rep.rep.delta_log else '<div class="anom-label">NO ACTIONS YET</div>',
                    unsafe_allow_html=True)

    with st.expander("[ REPUTATION ACTION LOG ]"):
        rep_df = rep.rep_history_df()
        if not rep_df.empty:
            st.dataframe(rep_df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="anom-label">NO REPUTATION EVENTS RECORDED.</div>',
                        unsafe_allow_html=True)

    # ── Entity profile reference ───────────────────────────────────────────
    with st.expander("[ ENTITY CLASSIFICATION PROFILES ]"):
        entity_rows = []
        for entity, profile in EntityClassifier.ENTITY_PROFILES.items():
            entity_rows.append({
                "Entity":      entity.value,
                "Description": profile["desc"],
                "WoW Range":   f"{profile['wow_range'][0]}–{profile['wow_range'][1]}",
                "Anom Range":  f"{profile['anomaly_range'][0]}–{profile['anomaly_range'][1]}",
                "Prime p<":    profile["prime_test_max"],
                "Fib R²≥":     profile["fib_r2_min"],
                "Periodic":    "✓" if profile["periodicity"] else "—",
                "Threat Max":  profile["threat_max"].name,
                "Rep Effect":  f"{profile.get('rep_on_detect', 0):+d}",
            })
        st.dataframe(pd.DataFrame(entity_rows),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    anomaly_detector_page()
