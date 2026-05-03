"""
hq_reporter.py — Headquarters Reporting & Data Submission Engine
Voices of the Void | Alpen Signal Observatorium — Dunkeltaler Forest
Secure Transmission & Archive Management System v3.7.2

Handles: Cryptographic drive hashing and verification, HQ submission protocol,
         signal archive with full FITS-style metadata, point scoring engine,
         leaderboard mechanics, transmission scheduling, uplink simulation,
         data integrity checking (Reed-Solomon proxy), session report generation,
         compressed archive packaging, signal deduplication, priority queuing,
         HQ response parsing, badge / achievement system, anomaly escalation
         routing, ARIRAL rep integration with HQ bulletin.

All data that leaves this observatory goes through this module.
All data. Every byte. Verified. Signed. Logged.
If you bypass this module, you are lying to HQ.
They will know.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import io
import json
import math
import os
import struct
import time
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# STATION & HQ CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

STATION_ID          = "ASO-DUNKELTALER-01"
HQ_CALL_SIGN        = "HQ-PROMETHEUS-CENTRAL"
PROTOCOL_VERSION    = "VoV-PROTO-3.7"
SECRET_KEY          = b"dunkeltaler_void_obs_2024_CLASSIFIED"
SUBMISSION_INTERVAL_H = 24.0
UPLINK_BANDWIDTH_MBPS = 0.128         # satellite uplink (128 kbps)
MAX_PAYLOAD_MB        = 512.0
REED_SOLOMON_OVERHEAD = 0.25          # 25% error correction overhead
COMPRESSION_RATIO_EST = 0.45         # estimated gzip compression

# Signal value multipliers per class (base points)
SIGNAL_BASE_POINTS: Dict[str, int] = {
    "NARROWBAND_CW":       50,
    "NARROWBAND_PULSED":   75,
    "PULSAR":              200,
    "CHIRP":               100,
    "BROADBAND_BURST":     180,
    "STRUCTURED_BPSK":     300,
    "STRUCTURED_FSK":      280,
    "ASTROPHYSICAL_LINE":  120,
    "ANOMALOUS":           400,
    "ARIRAL":              800,
    "VOID_CARRIER":        0,          # NEVER submit; erased
}

WOW_MULTIPLIERS: List[Tuple[float, float]] = [
    (0.0,  2.0,  1.0),
    (2.0,  4.0,  1.5),
    (4.0,  6.0,  2.5),
    (6.0,  8.0,  4.0),
    (8.0,  10.0, 8.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class SubmissionStatus(Enum):
    PENDING     = "PENDING"
    QUEUED      = "QUEUED"
    TRANSMITTING = "TRANSMITTING"
    DELIVERED   = "DELIVERED"
    REJECTED    = "REJECTED"
    CORRUPTED   = "CORRUPTED"

class HQResponseCode(Enum):
    ACK_CLEAN         = "ACK_CLEAN"
    ACK_WITH_WARNINGS = "ACK_WITH_WARNINGS"
    REJECT_INTEGRITY  = "REJECT_INTEGRITY"
    REJECT_DUPLICATE  = "REJECT_DUPLICATE"
    REJECT_VOID_CONT  = "REJECT_VOID_CONT"
    PRIORITY_ESCALATE = "PRIORITY_ESCALATE"
    ARIRAL_COMMEND    = "ARIRAL_COMMEND"

class AchievementID(Enum):
    FIRST_SIGNAL          = "FIRST_SIGNAL"
    WOW_7_PLUS            = "WOW_7_PLUS"
    PULSAR_CONFIRMED      = "PULSAR_CONFIRMED"
    ARIRAL_FRIENDLY       = "ARIRAL_FRIENDLY"
    ARIRAL_ALLIED         = "ARIRAL_ALLIED"
    DRIVE_STREAK_5        = "DRIVE_STREAK_5"
    ANOMALY_HUNTER        = "ANOMALY_HUNTER"
    NIGHT_OWL             = "NIGHT_OWL"
    VOID_ERASED           = "VOID_ERASED"
    LOOKER_SURVIVED       = "LOOKER_SURVIVED"
    PERFECT_DRIVE         = "PERFECT_DRIVE"
    CENTURY_SIGNALS       = "CENTURY_SIGNALS"
    WEEK_SURVIVAL         = "WEEK_SURVIVAL"
    HYDROGEN_LINE_EXACT   = "HYDROGEN_LINE_EXACT"

class TransmissionPriority(Enum):
    ROUTINE   = 0
    EXPEDITED = 1
    PRIORITY  = 2
    EMERGENCY = 3

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalArchiveRecord:
    uid:                str
    timestamp_iso:      str
    signal_class:       str
    origin:             str
    center_freq_mhz:    float
    bandwidth_hz:       float
    drift_rate_hz_s:    float
    snr_db:             float
    dispersion_measure: float
    ra_hours:           float
    dec_degrees:        float
    wow_factor:         float
    classifier_conf:    float
    anomaly_score:      float
    entity_class:       str
    threat_level:       str
    stage:              str
    hash_code:          str
    drive_id:           str
    day_number:         int
    ingame_time_h:      float
    notes:              str
    base_points:        int
    wow_multiplier:     float
    rep_bonus:          float
    final_points:       int
    integrity_hash:     str  = ""

    def __post_init__(self):
        if not self.integrity_hash:
            payload = (f"{self.uid}{self.timestamp_iso}{self.signal_class}"
                       f"{self.center_freq_mhz:.6f}{self.snr_db:.4f}{self.hash_code}")
            self.integrity_hash = hmac.new(
                SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()[:24].upper()

    def verify_integrity(self) -> bool:
        payload = (f"{self.uid}{self.timestamp_iso}{self.signal_class}"
                   f"{self.center_freq_mhz:.6f}{self.snr_db:.4f}{self.hash_code}")
        expected = hmac.new(SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()[:24].upper()
        return hmac.compare_digest(self.integrity_hash, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriveSubmissionPackage:
    package_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:12].upper())
    drive_id:         str   = ""
    station_id:       str   = STATION_ID
    protocol_version: str   = PROTOCOL_VERSION
    created_at:       float = field(default_factory=time.time)
    records:          List[SignalArchiveRecord] = field(default_factory=list)
    drive_hash:       str   = ""
    package_hash:     str   = ""
    total_signals:    int   = 0
    total_points:     int   = 0
    ariral_signals:   int   = 0
    anomalous_signals: int  = 0
    wow_max:          float = 0.0
    status:           SubmissionStatus = SubmissionStatus.PENDING
    hq_response:      Optional[str]    = None
    transmission_time_s: float = 0.0
    payload_mb:          float = 0.0
    compressed_mb:       float = 0.0
    rs_overhead_mb:      float = 0.0
    total_on_wire_mb:    float = 0.0
    day_submitted:       int   = 0

    def compute_hashes(self) -> None:
        self.total_signals    = len(self.records)
        self.total_points     = sum(r.final_points for r in self.records)
        self.ariral_signals   = sum(1 for r in self.records if r.signal_class == "ARIRAL")
        self.anomalous_signals = sum(1 for r in self.records if r.signal_class in ("ANOMALOUS","ARIRAL"))
        self.wow_max          = max((r.wow_factor for r in self.records), default=0.0)
        # Drive hash: HMAC-SHA256 of all record UIDs
        uid_concat = "|".join(sorted(r.uid for r in self.records))
        self.drive_hash  = hmac.new(SECRET_KEY, uid_concat.encode(),
                                     hashlib.sha256).hexdigest().upper()
        # Package hash: hash of metadata
        meta = (f"{self.package_id}{self.drive_id}{self.station_id}"
                f"{self.total_signals}{self.total_points}{self.drive_hash}")
        self.package_hash = hmac.new(SECRET_KEY, meta.encode(),
                                      hashlib.sha256).hexdigest().upper()

    def simulate_transmission(self) -> Tuple[float, float, float, float]:
        """Simulate realistic uplink. Returns (raw_mb, compressed_mb, rs_mb, total_mb)."""
        raw_bytes = sum(len(json.dumps(r.to_dict()).encode()) for r in self.records)
        raw_mb   = raw_bytes / 1e6
        comp_mb  = raw_mb * COMPRESSION_RATIO_EST
        rs_mb    = comp_mb * (1 + REED_SOLOMON_OVERHEAD)
        tx_time  = rs_mb * 8 / UPLINK_BANDWIDTH_MBPS  # seconds
        self.payload_mb       = raw_mb
        self.compressed_mb    = comp_mb
        self.rs_overhead_mb   = rs_mb - comp_mb
        self.total_on_wire_mb = rs_mb
        self.transmission_time_s = tx_time
        return raw_mb, comp_mb, rs_mb, tx_time


@dataclass
class HQResponse:
    package_id:    str
    response_code: HQResponseCode
    timestamp:     float = field(default_factory=time.time)
    verified_count: int   = 0
    rejected_count: int   = 0
    bonus_points:   int   = 0
    rep_delta:      int   = 0
    bulletin:       str   = ""
    priority_flag:  bool  = False
    achievement_ids: List[str] = field(default_factory=list)


@dataclass
class Achievement:
    achievement_id: AchievementID
    title:          str
    description:    str
    points_reward:  int
    unlocked:       bool  = False
    unlocked_at:    float = 0.0
    progress:       float = 0.0     # 0–1

    def unlock(self) -> None:
        if not self.unlocked:
            self.unlocked    = True
            self.unlocked_at = time.time()
            self.progress    = 1.0


@dataclass
class SessionStats:
    session_id:         str   = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    start_time:         float = field(default_factory=time.time)
    day_number:         int   = 1
    total_points:       int   = 0
    total_signals:      int   = 0
    drives_submitted:   int   = 0
    drives_erased:      int   = 0
    anomaly_count:      int   = 0
    ariral_count:       int   = 0
    void_carrier_count: int   = 0
    highest_wow:        float = 0.0
    highest_snr:        float = 0.0
    rep_current:        int   = 0
    achievements_unlocked: List[str] = field(default_factory=list)
    signal_class_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    points_by_day:       Dict[int, int]  = field(default_factory=lambda: defaultdict(int))
    hourly_signal_rate:  List[float]     = field(default_factory=list)

    @property
    def uptime_h(self) -> float:
        return (time.time() - self.start_time) / 3600

    @property
    def signals_per_hour(self) -> float:
        up = self.uptime_h
        return self.total_signals / max(up, 0.01)

    @property
    def points_per_signal(self) -> float:
        return self.total_points / max(self.total_signals, 1)

    @property
    def anomaly_rate(self) -> float:
        return self.anomaly_count / max(self.total_signals, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes points for each submitted signal record.
    Factors: base class value × WoW multiplier × SNR bonus × rep bonus
             × novelty bonus × integrity bonus.
    """

    @staticmethod
    def wow_multiplier(wow: float) -> float:
        for lo, hi, mult in WOW_MULTIPLIERS:
            if lo <= wow < hi:
                return mult
        return WOW_MULTIPLIERS[-1][2]

    @staticmethod
    def snr_bonus(snr_db: float) -> float:
        """Additional multiplier for very high SNR signals."""
        if snr_db >= 30:  return 1.5
        if snr_db >= 20:  return 1.2
        if snr_db >= 10:  return 1.0
        return 0.8

    @staticmethod
    def novelty_bonus(signal_class: str, seen_classes: set) -> float:
        """First time seeing this class in session: +50%."""
        if signal_class not in seen_classes:
            return 1.5
        return 1.0

    @staticmethod
    def rep_bonus(rep_value: int) -> float:
        """ARIRAL reputation multiplier on ARIRAL-class signals."""
        if rep_value >= 50:   return 1.6
        if rep_value >= 10:   return 1.3
        if rep_value >= -10:  return 1.0
        if rep_value >= -50:  return 0.8
        return 0.5

    @staticmethod
    def integrity_bonus(record_verified: bool) -> float:
        return 1.1 if record_verified else 0.9

    def compute(self, signal_class: str, wow: float, snr_db: float,
                rep: int, seen_classes: set, integrity_ok: bool) -> Tuple[int, float, float]:
        """
        Returns (final_points, wow_mult, rep_mult).
        """
        base    = SIGNAL_BASE_POINTS.get(signal_class, 50)
        w_mult  = self.wow_multiplier(wow)
        s_bonus = self.snr_bonus(snr_db)
        n_bonus = self.novelty_bonus(signal_class, seen_classes)
        r_mult  = self.rep_bonus(rep)
        i_bonus = self.integrity_bonus(integrity_ok)
        final   = int(base * w_mult * s_bonus * n_bonus * r_mult * i_bonus)
        return final, w_mult, r_mult


# ─────────────────────────────────────────────────────────────────────────────
# ARCHIVE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ArchiveManager:
    """
    Maintains the full session signal archive with deduplication,
    integrity verification, priority queuing, and retrieval.
    """

    def __init__(self):
        self._records:       List[SignalArchiveRecord] = []
        self._uid_index:     Dict[str, int] = {}         # uid -> list index
        self._hash_seen:     set = set()                 # for deduplication
        self._priority_queue: deque = deque()            # (priority, uid) pairs
        self._scoring = ScoringEngine()
        self._seen_classes: set = set()
        self._n_integrity_failures = 0

    def ingest(self, signal_dict: Dict[str, Any],
               rep_value: int = 0,
               day: int = 1,
               ingame_time: float = 12.0) -> Optional[SignalArchiveRecord]:
        """
        Ingest a signal record dict (from SignalRecord.to_dataframe_row()).
        Returns SignalArchiveRecord if accepted, None if duplicate or VOID_CARRIER.
        """
        # Reject VOID_CARRIER class immediately
        sig_class = signal_dict.get("Class", signal_dict.get("signal_class", "UNKNOWN"))
        if "VOID" in sig_class:
            return None

        # Deduplication by hash_code
        hash_code = signal_dict.get("Hash", signal_dict.get("hash_code", ""))
        if hash_code in self._hash_seen:
            return None
        self._hash_seen.add(hash_code)

        uid = signal_dict.get("UID", str(uuid.uuid4())[:12].upper())
        wow = float(signal_dict.get("WoW", signal_dict.get("wow_factor", 0.0)))
        snr = float(signal_dict.get("SNR(dB)", signal_dict.get("snr_db", 0.0)))

        # Scoring
        final_pts, w_mult, r_mult = self._scoring.compute(
            sig_class, wow, snr, rep_value, self._seen_classes, integrity_ok=True)
        self._seen_classes.add(sig_class)

        # Build archive record
        rec = SignalArchiveRecord(
            uid=uid,
            timestamp_iso=pd.Timestamp(time.time(), unit="s").isoformat(),
            signal_class=sig_class,
            origin=signal_dict.get("Origin", signal_dict.get("origin", "UNKNOWN")),
            center_freq_mhz=float(signal_dict.get("Freq(MHz)", signal_dict.get("center_freq_mhz", 1420.0))),
            bandwidth_hz=float(signal_dict.get("BW(Hz)", signal_dict.get("bandwidth_hz", 1000.0))),
            drift_rate_hz_s=float(signal_dict.get("Drift(Hz/s)", signal_dict.get("drift_rate_hz_s", 0.0))),
            snr_db=snr,
            dispersion_measure=float(signal_dict.get("DM", signal_dict.get("dispersion_measure", 0.0))),
            ra_hours=float(signal_dict.get("RA(h)", signal_dict.get("ra_hours", 0.0))),
            dec_degrees=float(signal_dict.get("Dec(°)", signal_dict.get("dec_degrees", 0.0))),
            wow_factor=wow,
            classifier_conf=float(signal_dict.get("Conf%", signal_dict.get("classifier_confidence", 0.0))),
            anomaly_score=float(signal_dict.get("anomaly_score", 0.0)),
            entity_class=signal_dict.get("entity_class", "NONE"),
            threat_level=signal_dict.get("Threat", signal_dict.get("threat_level", "NOMINAL")),
            stage=signal_dict.get("Stage", signal_dict.get("stage", "CLASSIFIED")),
            hash_code=hash_code,
            drive_id=signal_dict.get("Drive", "UNASSIGNED"),
            day_number=day,
            ingame_time_h=ingame_time,
            notes=signal_dict.get("notes", ""),
            base_points=SIGNAL_BASE_POINTS.get(sig_class, 50),
            wow_multiplier=w_mult,
            rep_bonus=r_mult,
            final_points=final_pts,
        )

        # Integrity check
        if not rec.verify_integrity():
            self._n_integrity_failures += 1

        idx = len(self._records)
        self._records.append(rec)
        self._uid_index[uid] = idx

        # Priority queue entry
        priority = TransmissionPriority.EMERGENCY if wow >= 8 else \
                   TransmissionPriority.PRIORITY   if wow >= 6 else \
                   TransmissionPriority.EXPEDITED  if wow >= 4 else \
                   TransmissionPriority.ROUTINE
        self._priority_queue.append((priority.value, uid))

        return rec

    def build_drive_package(self, drive_id: str,
                             max_records: int = 200) -> DriveSubmissionPackage:
        """
        Build a submission package from pending records.
        Takes highest-priority records up to max_records.
        """
        # Sort queue by priority descending
        sorted_q = sorted(self._priority_queue, key=lambda x: -x[0])
        selected_uids = set()
        for _, uid in sorted_q:
            if len(selected_uids) >= max_records:
                break
            selected_uids.add(uid)

        records = [self._records[self._uid_index[uid]]
                   for uid in selected_uids
                   if uid in self._uid_index]

        pkg = DriveSubmissionPackage(drive_id=drive_id, records=records)
        pkg.compute_hashes()
        pkg.simulate_transmission()
        return pkg

    def get_all(self) -> List[SignalArchiveRecord]:
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = []
        for r in self._records:
            rows.append({
                "UID":      r.uid,
                "Day":      r.day_number,
                "Class":    r.signal_class,
                "Origin":   r.origin,
                "Freq(MHz)":round(r.center_freq_mhz, 4),
                "SNR(dB)":  round(r.snr_db, 1),
                "WoW":      round(r.wow_factor, 2),
                "Threat":   r.threat_level,
                "Entity":   r.entity_class,
                "Points":   r.final_points,
                "WowMult":  round(r.wow_multiplier, 1),
                "RepMult":  round(r.rep_bonus, 1),
                "DM":       round(r.dispersion_measure, 1),
                "Verified": "✓" if r.verify_integrity() else "✗",
                "Hash":     r.integrity_hash,
            })
        return pd.DataFrame(rows)

    def stats_by_class(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        counts:  Dict[str, int]   = defaultdict(int)
        points:  Dict[str, int]   = defaultdict(int)
        wow_max: Dict[str, float] = defaultdict(float)
        for r in self._records:
            counts[r.signal_class]  += 1
            points[r.signal_class]  += r.final_points
            wow_max[r.signal_class]  = max(wow_max[r.signal_class], r.wow_factor)
        rows = []
        for cls in sorted(counts.keys()):
            rows.append({
                "Class":       cls,
                "Count":       counts[cls],
                "Total Points":points[cls],
                "Avg Points":  round(points[cls] / counts[cls]),
                "Max WoW":     round(wow_max[cls], 2),
                "Base Val":    SIGNAL_BASE_POINTS.get(cls, 50),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HQ COMMUNICATION PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class HQProtocol:
    """
    Simulates the HQ uplink protocol.
    In a real deployment this would be a satellite/TCP connection.
    Here we simulate HQ responses with realistic latency and rejection logic.
    """

    HQ_BULLETINS = [
        "ROUTINE: Observation season proceeding nominally. Continue scheduled recordings.",
        "INFO: Geomagnetic Kp index elevated this week. Expect ionospheric degradation.",
        "PRIORITY: ARIRAL signal cluster detected by array B-7. Cross-reference your data.",
        "WARNING: Do not transmit VOID class signals. Erase immediately. Protocol 7-OMEGA.",
        "CLASSIFIED: Report any prime-encoded signals above WoW 8.5 via secure channel.",
        "INFO: Supply drop rescheduled to +3 days. Ration remaining food stocks.",
        "ROUTINE: Satellite uplink window expanded by 2h this cycle. Extended transmit OK.",
        "ALERT: Unidentified object detected approaching array perimeter. Stay inside.",
        "INFO: ARIRAL reputation above FRIENDLY tier noted. Commendation forwarded.",
        "WARNING: This station's anomaly rate is 2.3σ above network mean. Verify equipment.",
        "PRIORITY: Looker event sequence reported by three stations. Remain calm. Document.",
        "INFO: New signal classification schema update. Please retrain your local model.",
        "CLASSIFIED: Do not discuss 03:33 events on open channels. Use cipher K-9.",
    ]

    def submit(self, package: DriveSubmissionPackage,
               inject_error: bool = False) -> HQResponse:
        """
        Simulate HQ package submission and response.
        """
        # Simulate transmission delay
        tx_time = package.transmission_time_s

        # Integrity verification by HQ
        verified_count = sum(1 for r in package.records if r.verify_integrity())
        rejected_count = len(package.records) - verified_count

        # Determine response code
        rng = np.random.default_rng(int(package.package_id[:4], 16) % (2**31))
        if inject_error or rejected_count > len(package.records) * 0.2:
            code = HQResponseCode.REJECT_INTEGRITY
        elif any(r.signal_class == "VOID_CARRIER" for r in package.records):
            code = HQResponseCode.REJECT_VOID_CONT
        elif package.wow_max >= 8.0:
            code = HQResponseCode.PRIORITY_ESCALATE
        elif package.ariral_signals > 0:
            code = HQResponseCode.ARIRAL_COMMEND
        elif rng.random() < 0.05:
            code = HQResponseCode.ACK_WITH_WARNINGS
        else:
            code = HQResponseCode.ACK_CLEAN

        # Bonus points from HQ
        bonus = 0
        if code == HQResponseCode.ACK_CLEAN:
            bonus = int(package.total_points * 0.10)
        elif code == HQResponseCode.ARIRAL_COMMEND:
            bonus = int(package.total_points * 0.25)
        elif code == HQResponseCode.PRIORITY_ESCALATE:
            bonus = int(package.total_points * 0.40)

        # Rep delta
        rep_delta = 0
        if code == HQResponseCode.ARIRAL_COMMEND:
            rep_delta = +10
        elif code == HQResponseCode.REJECT_VOID_CONT:
            rep_delta = -50
        elif code == HQResponseCode.ACK_CLEAN:
            rep_delta = +2

        # Achievements triggered
        achievement_ids: List[str] = []
        if package.wow_max >= 7.0:
            achievement_ids.append(AchievementID.WOW_7_PLUS.value)
        if package.ariral_signals >= 3:
            achievement_ids.append(AchievementID.ARIRAL_FRIENDLY.value)
        if verified_count == len(package.records) and len(package.records) >= 10:
            achievement_ids.append(AchievementID.PERFECT_DRIVE.value)

        bulletin = str(rng.choice(self.HQ_BULLETINS))

        package.status     = (SubmissionStatus.DELIVERED
                               if code in (HQResponseCode.ACK_CLEAN,
                                           HQResponseCode.ACK_WITH_WARNINGS,
                                           HQResponseCode.PRIORITY_ESCALATE,
                                           HQResponseCode.ARIRAL_COMMEND)
                               else SubmissionStatus.REJECTED)
        package.hq_response = code.value

        return HQResponse(
            package_id=package.package_id,
            response_code=code,
            verified_count=verified_count,
            rejected_count=rejected_count,
            bonus_points=bonus,
            rep_delta=rep_delta,
            bulletin=bulletin,
            priority_flag=code == HQResponseCode.PRIORITY_ESCALATE,
            achievement_ids=achievement_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ACHIEVEMENT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class AchievementSystem:
    """Full achievement/badge system tracking session progress."""

    ACHIEVEMENT_DEFS: Dict[AchievementID, Tuple[str, str, int]] = {
        AchievementID.FIRST_SIGNAL:         ("First Contact",            "Record your first signal.",          100),
        AchievementID.WOW_7_PLUS:           ("Extraordinary Signal",     "Detect a signal with WoW ≥ 7.",     500),
        AchievementID.PULSAR_CONFIRMED:     ("Pulsar Hunter",            "Confirm a pulsar candidate.",        400),
        AchievementID.ARIRAL_FRIENDLY:      ("Trusted by the Void",      "Reach ARIRAL FRIENDLY tier.",        300),
        AchievementID.ARIRAL_ALLIED:        ("Voice of the Void",        "Reach ARIRAL ALLIED tier.",          1000),
        AchievementID.DRIVE_STREAK_5:       ("Reliable Reporter",        "Submit 5 drives in a row.",          250),
        AchievementID.ANOMALY_HUNTER:       ("Pattern Recognition",      "Detect 10 anomalous signals.",       350),
        AchievementID.NIGHT_OWL:            ("03:33",                    "Record a signal during the 03:33 window.", 200),
        AchievementID.VOID_ERASED:          ("Protocol 7-OMEGA",         "Correctly erase a VOID_CARRIER drive.", 600),
        AchievementID.LOOKER_SURVIVED:      ("You Weren't Alone",        "Survive a Looker event sequence.",   750),
        AchievementID.PERFECT_DRIVE:        ("Clean Transmission",       "Submit a drive with 100% integrity.", 400),
        AchievementID.CENTURY_SIGNALS:      ("Century Club",             "Record 100 signals.",                 500),
        AchievementID.WEEK_SURVIVAL:        ("One Week In",              "Survive 7 days at the station.",     800),
        AchievementID.HYDROGEN_LINE_EXACT:  ("21-Centimetre Resonance",  "Detect signal at exact H-I frequency.", 350),
    }

    def __init__(self):
        self._achievements: Dict[AchievementID, Achievement] = {
            aid: Achievement(
                achievement_id=aid,
                title=defs[0],
                description=defs[1],
                points_reward=defs[2],
            )
            for aid, defs in self.ACHIEVEMENT_DEFS.items()
        }

    def check_and_unlock(self, stats: SessionStats,
                          archive: ArchiveManager,
                          rep_value: int,
                          recent_event: Optional[str] = None) -> List[Achievement]:
        """
        Check all achievement conditions and unlock as appropriate.
        Returns list of newly unlocked achievements.
        """
        newly_unlocked: List[Achievement] = []

        def _unlock(aid: AchievementID) -> None:
            ach = self._achievements[aid]
            if not ach.unlocked:
                ach.unlock()
                stats.achievements_unlocked.append(aid.value)
                stats.total_points += ach.points_reward
                newly_unlocked.append(ach)

        records = archive.get_all()

        if stats.total_signals >= 1:
            _unlock(AchievementID.FIRST_SIGNAL)
        if stats.highest_wow >= 7.0:
            _unlock(AchievementID.WOW_7_PLUS)
        if any(r.signal_class == "PULSAR" for r in records):
            _unlock(AchievementID.PULSAR_CONFIRMED)
        if rep_value >= 10:
            _unlock(AchievementID.ARIRAL_FRIENDLY)
        if rep_value >= 50:
            _unlock(AchievementID.ARIRAL_ALLIED)
        if stats.drives_submitted >= 5:
            _unlock(AchievementID.DRIVE_STREAK_5)
        if stats.anomaly_count >= 10:
            _unlock(AchievementID.ANOMALY_HUNTER)
        if stats.void_carrier_count >= 1:
            _unlock(AchievementID.VOID_ERASED)
        if stats.total_signals >= 100:
            _unlock(AchievementID.CENTURY_SIGNALS)
        if stats.day_number >= 7:
            _unlock(AchievementID.WEEK_SURVIVAL)
        if any(abs(r.center_freq_mhz - 1420.405751768) < 0.001 for r in records):
            _unlock(AchievementID.HYDROGEN_LINE_EXACT)
        if recent_event == "LOOKER_SURVIVED":
            _unlock(AchievementID.LOOKER_SURVIVED)
        if recent_event == "VOID_ERASED":
            _unlock(AchievementID.VOID_ERASED)
        if recent_event == "03:33":
            _unlock(AchievementID.NIGHT_OWL)

        return newly_unlocked

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for aid, ach in self._achievements.items():
            rows.append({
                "Badge":       ach.title,
                "Description": ach.description,
                "Points":      ach.points_reward,
                "Status":      "✓ UNLOCKED" if ach.unlocked else "— LOCKED",
                "Unlocked At": (pd.Timestamp(ach.unlocked_at, unit="s").strftime("%H:%M:%S")
                                if ach.unlocked else "—"),
            })
        return pd.DataFrame(rows)

    def unlocked_count(self) -> int:
        return sum(1 for a in self._achievements.values() if a.unlocked)

    def total_achievement_points(self) -> int:
        return sum(a.points_reward for a in self._achievements.values() if a.unlocked)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates human-readable session reports and HQ summaries.
    Format mirrors VotV's in-game terminal aesthetic.
    """

    HEADER = "=" * 62
    SUBHDR = "-" * 62

    def generate_session_report(self, stats: SessionStats,
                                  archive: ArchiveManager,
                                  achievements: AchievementSystem,
                                  packages: List[DriveSubmissionPackage],
                                  rep_value: int) -> str:
        lines: List[str] = []
        lines.append(self.HEADER)
        lines.append(f"  ALPEN SIGNAL OBSERVATORIUM — SESSION REPORT")
        lines.append(f"  Station: {STATION_ID}")
        lines.append(f"  Session: {stats.session_id}")
        lines.append(f"  Day: {stats.day_number} | Uptime: {stats.uptime_h:.1f}h")
        lines.append(self.HEADER)
        lines.append("")
        lines.append("[SIGNAL STATISTICS]")
        lines.append(self.SUBHDR)
        lines.append(f"  Total Signals Recorded : {stats.total_signals:>8}")
        lines.append(f"  Anomalous Signals       : {stats.anomaly_count:>8}")
        lines.append(f"  ARIRAL Signals          : {stats.ariral_count:>8}")
        lines.append(f"  VOID CARRIER Events     : {stats.void_carrier_count:>8}")
        lines.append(f"  Highest WoW Score       : {stats.highest_wow:>8.3f}")
        lines.append(f"  Highest SNR             : {stats.highest_snr:>8.1f} dB")
        lines.append(f"  Signals/Hour            : {stats.signals_per_hour:>8.2f}")
        lines.append(f"  Anomaly Rate            : {stats.anomaly_rate*100:>7.1f}%")
        lines.append("")
        lines.append("[SCORING]")
        lines.append(self.SUBHDR)
        lines.append(f"  Total Points            : {stats.total_points:>8}")
        lines.append(f"  Points / Signal         : {stats.points_per_signal:>8.1f}")
        lines.append(f"  Drives Submitted        : {stats.drives_submitted:>8}")
        lines.append(f"  Drives Erased           : {stats.drives_erased:>8}")
        lines.append(f"  Achievement Points      : {achievements.total_achievement_points():>8}")
        lines.append(f"  Achievements Unlocked   : {achievements.unlocked_count():>8} / {len(achievements._achievements)}")
        lines.append("")
        lines.append("[ARIRAL REPUTATION]")
        lines.append(self.SUBHDR)
        lines.append(f"  Current Rep             : {rep_value:>+8}")
        rep_state = ("ALLIED" if rep_value >= 50 else "FRIENDLY" if rep_value >= 10 else
                     "NEUTRAL" if rep_value >= -10 else "WARY" if rep_value >= -50 else "HOSTILE")
        lines.append(f"  Reputation Tier         : {rep_state:>8}")
        lines.append("")
        lines.append("[SIGNAL CLASS BREAKDOWN]")
        lines.append(self.SUBHDR)
        class_df = archive.stats_by_class()
        if not class_df.empty:
            for _, row in class_df.iterrows():
                lines.append(f"  {row['Class']:<22} : {row['Count']:>4} signals | "
                              f"{row['Total Points']:>7} pts | WoW max {row['Max WoW']:.2f}")
        else:
            lines.append("  No signals archived.")
        lines.append("")
        lines.append("[DRIVE SUBMISSIONS]")
        lines.append(self.SUBHDR)
        for pkg in packages[-10:]:
            status = pkg.status.value if pkg.status else "UNKNOWN"
            lines.append(f"  {pkg.drive_id:<14} | {pkg.total_signals:>3} sigs | "
                          f"{pkg.total_points:>6} pts | {pkg.compressed_mb:.2f} MB | "
                          f"{status}")
        lines.append("")
        lines.append("[ACHIEVEMENTS]")
        lines.append(self.SUBHDR)
        unlocked = [a for a in achievements._achievements.values() if a.unlocked]
        if unlocked:
            for ach in unlocked:
                lines.append(f"  [{ach.title}] — {ach.description} (+{ach.points_reward} pts)")
        else:
            lines.append("  No achievements unlocked.")
        lines.append("")
        lines.append(self.HEADER)
        lines.append(f"  END OF REPORT — {pd.Timestamp(time.time(), unit='s').isoformat()}")
        lines.append(self.HEADER)
        return "\n".join(lines)

    def generate_hq_bulletin(self, response: HQResponse) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║         HQ TRANSMISSION RECEIVED — PROMETHEUS CENTRAL   ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Package : {response.package_id:<47}║",
            f"║  Status  : {response.response_code.value:<47}║",
            f"║  Verified: {response.verified_count:<5}  Rejected: {response.rejected_count:<5}"
            f"  Bonus: +{response.bonus_points:<6}pts  ║",
            f"║  Rep Δ   : {response.rep_delta:+d:<47}║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  BULLETIN:                                               ║",
        ]
        bull = response.bulletin
        for i in range(0, len(bull), 55):
            chunk = bull[i:i+55]
            lines.append(f"║  {chunk:<57}║")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD (SESSION-SCOPED)
# ─────────────────────────────────────────────────────────────────────────────

class Leaderboard:
    """
    Session-scoped leaderboard tracking per-day performance.
    In a real multi-station deployment this would sync across stations.
    """

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def record_day(self, day: int, points: int, signals: int,
                    anomalies: int, rep: int, wow_max: float) -> None:
        self._entries.append({
            "Day":      day,
            "Points":   points,
            "Signals":  signals,
            "Anomalies":anomalies,
            "Rep":      rep,
            "WoW Max":  round(wow_max, 2),
            "Rank":     "—",
        })

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(columns=["Day","Points","Signals","Anomalies","Rep","WoW Max"])
        df = pd.DataFrame(self._entries)
        df = df.sort_values("Points", ascending=False).reset_index(drop=True)
        df.index += 1
        df["Rank"] = df.index
        return df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ─────────────────────────────────────────────────────────────────────────────

class HQVisualizer:
    PAL = {
        "bg":   "#030d07", "fg":   "#00ff88", "dim":  "#004422",
        "acc":  "#00ffcc", "warn": "#ff8800", "dng":  "#ff2222",
        "grid": "#001a0e", "axis": "#335544", "gold": "#ffcc00",
        "blue": "#0088ff", "purp": "#8844ff",
    }

    def _style(self, ax, title=""):
        ax.set_facecolor(self.PAL["bg"])
        for sp in ax.spines.values(): sp.set_edgecolor(self.PAL["dim"])
        ax.tick_params(colors=self.PAL["axis"], labelsize=6)
        ax.grid(True, color=self.PAL["grid"], alpha=0.4, ls=":")
        if title:
            ax.set_title(title, color=self.PAL["fg"], fontsize=7.5,
                         loc="left", pad=3, fontfamily="monospace")

    def plot_points_timeline(self, stats: SessionStats,
                              archive: ArchiveManager,
                              figsize=(10, 3.5)) -> plt.Figure:
        records = archive.get_all()
        if not records:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=12)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        pts   = [r.final_points for r in records]
        wow   = [r.wow_factor   for r in records]
        cumul = np.cumsum(pts)
        x     = np.arange(len(records))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        facecolor=self.PAL["bg"], sharex=True)
        # Cumulative points
        ax1.plot(x, cumul, color=self.PAL["fg"], lw=1.2)
        ax1.fill_between(x, cumul, 0, alpha=0.12, color=self.PAL["fg"])
        ax1.set_ylabel("Cumulative Points", fontsize=6, color=self.PAL["fg"])
        self._style(ax1, "POINTS ACCUMULATION")

        # Per-signal points coloured by WoW
        colors = [self.PAL["dng"]  if w >= 8 else
                   self.PAL["warn"] if w >= 5 else
                   self.PAL["fg"]   if w >= 2 else
                   self.PAL["dim"]
                   for w in wow]
        ax2.bar(x, pts, color=colors, alpha=0.8, width=0.85)
        ax2.set_xlabel("Signal Index", fontsize=6, color=self.PAL["fg"])
        ax2.set_ylabel("Points", fontsize=6, color=self.PAL["fg"])
        self._style(ax2, "PER-SIGNAL POINTS (colour = WoW tier)")

        patches = [
            mpatches.Patch(color=self.PAL["dim"],  label="WoW < 2"),
            mpatches.Patch(color=self.PAL["fg"],   label="WoW 2–5"),
            mpatches.Patch(color=self.PAL["warn"], label="WoW 5–8"),
            mpatches.Patch(color=self.PAL["dng"],  label="WoW ≥ 8"),
        ]
        ax2.legend(handles=patches, fontsize=5.5, loc="upper left",
                   facecolor=self.PAL["bg"], edgecolor=self.PAL["dim"],
                   labelcolor=self.PAL["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    def plot_class_distribution(self, archive: ArchiveManager,
                                 figsize=(9, 3.5)) -> plt.Figure:
        df = archive.stats_by_class()
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=12)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, (ax_cnt, ax_pts) = plt.subplots(1, 2, figsize=figsize,
                                              facecolor=self.PAL["bg"])
        classes = [c.replace("_", "\n") for c in df["Class"]]
        counts  = df["Count"].values
        pts     = df["Total Points"].values

        pal_cycle = [self.PAL["fg"], self.PAL["acc"], self.PAL["warn"],
                     self.PAL["blue"], self.PAL["purp"], self.PAL["gold"],
                     self.PAL["dng"], "#44ff44", "#ff44ff", "#00ffff"]

        for ax, vals, label, title in [
            (ax_cnt, counts, "Count",    "SIGNAL COUNT BY CLASS"),
            (ax_pts, pts,    "Points",   "TOTAL POINTS BY CLASS"),
        ]:
            bars = ax.barh(classes[::-1], vals[::-1],
                            color=[pal_cycle[i % len(pal_cycle)] for i in range(len(classes))][::-1],
                            alpha=0.82, height=0.65)
            for bar, val in zip(bars, vals[::-1]):
                ax.text(max(val * 0.02, 1), bar.get_y() + bar.get_height() / 2,
                         f"{val:,.0f}", va="center", fontsize=5.5,
                         color=self.PAL["acc"], fontfamily="monospace")
            self._style(ax, title)
            ax.set_xlabel(label, fontsize=6, color=self.PAL["fg"])

        plt.tight_layout(pad=0.4)
        return fig

    def plot_submission_history(self, packages: List[DriveSubmissionPackage],
                                 figsize=(10, 3.0)) -> plt.Figure:
        if not packages:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO SUBMISSIONS YET", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=11)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        x   = np.arange(len(packages))
        pts = [p.total_points for p in packages]
        sigs= [p.total_signals for p in packages]
        mb  = [p.total_on_wire_mb * 10 for p in packages]   # scaled

        cols = [self.PAL["fg"]   if p.status == SubmissionStatus.DELIVERED else
                self.PAL["warn"] if p.status == SubmissionStatus.PENDING else
                self.PAL["dng"]
                for p in packages]

        bars = ax.bar(x, pts, color=cols, alpha=0.82, width=0.7, label="Points")
        ax.plot(x, [s * max(pts) / max(sigs) for s in sigs],
                color=self.PAL["acc"], lw=0.9, marker="o", ms=3, label="Signals (scaled)")

        ax.set_xticks(x)
        ax.set_xticklabels([p.drive_id[-6:] for p in packages],
                            fontsize=5, color=self.PAL["axis"], rotation=30)
        self._style(ax, "DRIVE SUBMISSION HISTORY")
        ax.set_ylabel("Points", fontsize=6, color=self.PAL["fg"])
        ax.legend(fontsize=6, facecolor=self.PAL["bg"],
                  edgecolor=self.PAL["dim"], labelcolor=self.PAL["fg"])

        # Status annotations
        for bar, pkg in zip(bars, packages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pts) * 0.02,
                     pkg.status.value[0], ha="center", va="bottom",
                     fontsize=6, color=self.PAL["acc"], fontfamily="monospace")
        plt.tight_layout(pad=0.4)
        return fig

    def plot_achievement_grid(self, achievements: AchievementSystem,
                               figsize=(10, 5)) -> plt.Figure:
        all_ach = list(achievements._achievements.values())
        n   = len(all_ach)
        cols = 4
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                  facecolor=self.PAL["bg"])
        axes_flat = axes.flatten() if n > 1 else [axes]

        for i, ach in enumerate(all_ach):
            ax = axes_flat[i]
            ax.set_facecolor("#001a0e" if ach.unlocked else "#0a0a0a")
            for sp in ax.spines.values():
                sp.set_edgecolor(self.PAL["fg"] if ach.unlocked else self.PAL["dim"])
                sp.set_linewidth(1.5 if ach.unlocked else 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            col = self.PAL["fg"] if ach.unlocked else self.PAL["dim"]
            ax.text(0.5, 0.68, ach.title, transform=ax.transAxes,
                     ha="center", va="center", color=col,
                     fontsize=6, fontfamily="monospace", fontweight="bold",
                     wrap=True)
            ax.text(0.5, 0.38, f"+{ach.points_reward} pts",
                     transform=ax.transAxes, ha="center", va="center",
                     color=self.PAL["gold"] if ach.unlocked else self.PAL["dim"],
                     fontsize=7, fontfamily="monospace")
            status = "✓" if ach.unlocked else "—"
            ax.text(0.5, 0.12, status, transform=ax.transAxes,
                     ha="center", va="center", fontsize=9,
                     color=self.PAL["fg"] if ach.unlocked else self.PAL["dim"])

        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle("ACHIEVEMENT SYSTEM", color=self.PAL["fg"],
                      fontsize=9, fontfamily="monospace", y=1.01)
        plt.tight_layout(pad=0.3)
        return fig

    def plot_wow_distribution(self, archive: ArchiveManager,
                               figsize=(8, 3)) -> plt.Figure:
        records = archive.get_all()
        if not records:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=12)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        wows = [r.wow_factor for r in records]
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        _, bins, patches = ax.hist(wows, bins=20, range=(0, 10),
                                    color=self.PAL["fg"], alpha=0.75, edgecolor="none")
        # Colour by tier
        tier_edges = [0, 2, 4, 6, 8, 10]
        tier_cols  = [self.PAL["dim"], self.PAL["fg"], self.PAL["acc"],
                       self.PAL["warn"], self.PAL["dng"]]
        for patch, left in zip(patches, bins[:-1]):
            for i, edge in enumerate(tier_edges[:-1]):
                if edge <= left < tier_edges[i + 1]:
                    patch.set_facecolor(tier_cols[i])
                    break
        # Threshold lines
        for edge, label in [(2,""), (4,""), (6,""), (8,"")]:
            ax.axvline(edge, color=self.PAL["dim"], lw=0.6, ls=":", alpha=0.7)
        ax.axvline(float(np.mean(wows)), color=self.PAL["acc"], lw=1.0, ls="--",
                    label=f"Mean = {np.mean(wows):.2f}")
        self._style(ax, "WoW SCORE DISTRIBUTION")
        ax.set_xlabel("WoW Score", fontsize=6, color=self.PAL["fg"])
        ax.set_ylabel("Count", fontsize=6, color=self.PAL["fg"])
        ax.legend(fontsize=6, facecolor=self.PAL["bg"],
                  edgecolor=self.PAL["dim"], labelcolor=self.PAL["fg"])
        plt.tight_layout(pad=0.4)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def hq_reporter_page():
    from signal_engine import init_session_state

    init_session_state()

    st.markdown("""
    <style>
    .hq-header {
        font-family:'Courier New',monospace;
        color:#00ff88;
        font-size:0.78rem;
        letter-spacing:0.14em;
        border-bottom:1px solid #00ff4430;
        padding-bottom:0.4rem;
        margin-bottom:1rem;
    }
    .hq-label {
        font-family:'Courier New',monospace;
        color:#88ffcc;
        font-size:0.70rem;
        letter-spacing:0.09em;
        margin-top:0.5rem;
        margin-bottom:0.2rem;
    }
    .hq-bulletin {
        background:#001a0e;
        border:1px solid #00ff4440;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.68rem;
        color:#88ffcc;
        white-space:pre;
        overflow-x:auto;
        margin-bottom:0.6rem;
    }
    .hq-alert {
        background:#1a0500;
        border:1px solid #ff4400;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.70rem;
        color:#ff6622;
        margin-bottom:0.5rem;
    }
    .ach-new {
        background:#001a00;
        border:2px solid #ffcc00;
        border-radius:2px;
        padding:0.4rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.72rem;
        color:#ffcc00;
        margin-bottom:0.4rem;
    }
    .pts-box {
        background:#001a0e;
        border:1px solid #00ff44;
        border-radius:2px;
        padding:0.4rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.80rem;
        color:#00ff88;
        text-align:center;
        margin-bottom:0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hq-header">[ HQ REPORTING & DATA SUBMISSION ENGINE — PROMETHEUS CENTRAL UPLINK ]</div>',
                unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────────────────
    if "archive" not in st.session_state:
        st.session_state.archive = ArchiveManager()
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = SessionStats()
    if "achievements" not in st.session_state:
        st.session_state.achievements = AchievementSystem()
    if "hq_protocol" not in st.session_state:
        st.session_state.hq_protocol = HQProtocol()
    if "report_gen" not in st.session_state:
        st.session_state.report_gen = ReportGenerator()
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = Leaderboard()
    if "submitted_packages" not in st.session_state:
        st.session_state.submitted_packages = []
    if "hq_viz" not in st.session_state:
        st.session_state.hq_viz = HQVisualizer()

    archive: ArchiveManager        = st.session_state.archive
    stats:   SessionStats          = st.session_state.session_stats
    ach:     AchievementSystem     = st.session_state.achievements
    hq:      HQProtocol            = st.session_state.hq_protocol
    rgen:    ReportGenerator       = st.session_state.report_gen
    lb:      Leaderboard           = st.session_state.leaderboard
    pkgs:    List[DriveSubmissionPackage] = st.session_state.submitted_packages
    viz:     HQVisualizer          = st.session_state.hq_viz

    rep_value = int(getattr(
        getattr(st.session_state.get("rep_manager", None), "rep", None),
        "current", 0))
    dm:   Any = st.session_state.get("drive_manager", None)

    # ── Score totals bar ───────────────────────────────────────────────────
    st.markdown(f'<div class="pts-box">TOTAL POINTS: {stats.total_points:,} | '
                f'SIGNALS: {stats.total_signals} | '
                f'DAY: {stats.day_number} | '
                f'ACHIEVEMENTS: {ach.unlocked_count()}/{len(ach._achievements)}</div>',
                unsafe_allow_html=True)

    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown('<div class="hq-label">— INGEST SIGNAL LOG —</div>', unsafe_allow_html=True)
        if st.button("⬡ INGEST SESSION SIGNAL LOG", use_container_width=True):
            signal_log = st.session_state.get("signal_log", [])
            anomaly_recs = st.session_state.get("anomaly_records", [])
            ingested_n = 0

            for row in signal_log:
                rec = archive.ingest(row, rep_value=rep_value,
                                     day=int(st.session_state.day),
                                     ingame_time=12.0)
                if rec is not None:
                    ingested_n += 1
                    stats.total_signals += 1
                    stats.total_points  += rec.final_points
                    stats.points_by_day[stats.day_number] += rec.final_points
                    stats.signal_class_counts[rec.signal_class] += 1
                    stats.highest_wow = max(stats.highest_wow, rec.wow_factor)
                    stats.highest_snr = max(stats.highest_snr, rec.snr_db)
                    if rec.signal_class in ("ANOMALOUS", "ARIRAL"):
                        stats.anomaly_count += 1
                    if rec.signal_class == "ARIRAL":
                        stats.ariral_count += 1
                    if rec.signal_class == "VOID_CARRIER":
                        stats.void_carrier_count += 1

            # Ingest anomaly records for entity data
            for a_rec in anomaly_recs:
                pass  # already ingested via signal_log

            st.success(f"Ingested {ingested_n} new records | Total: {len(archive.get_all())}")

        st.markdown('<div class="hq-label">— DRIVE SUBMISSION —</div>', unsafe_allow_html=True)
        drive_id_input = st.text_input("Drive ID (leave blank for auto)", "")
        inject_error   = st.checkbox("Inject transmission error (test)", value=False)

        if st.button("⬡ BUILD & SUBMIT DRIVE PACKAGE", use_container_width=True):
            did = drive_id_input.strip() or f"DRV-{uuid.uuid4().hex[:6].upper()}"
            with st.spinner("[ BUILDING PACKAGE ... TRANSMITTING ... ]"):
                pkg = archive.build_drive_package(did, max_records=200)
                response = hq.submit(pkg, inject_error=inject_error)

                # Update stats
                stats.drives_submitted += 1
                stats.total_points += response.bonus_points
                pkgs.append(pkg)

                # Check achievements
                newly = ach.check_and_unlock(stats, archive, rep_value)

                # Update leaderboard
                lb.record_day(stats.day_number, stats.total_points,
                               stats.total_signals, stats.anomaly_count,
                               rep_value, stats.highest_wow)

            # Display bulletin
            bulletin_txt = rgen.generate_hq_bulletin(response)
            st.markdown(f'<div class="hq-bulletin">{bulletin_txt}</div>',
                        unsafe_allow_html=True)

            # Achievements
            if newly:
                for a in newly:
                    st.markdown(
                        f'<div class="ach-new">🏆 ACHIEVEMENT UNLOCKED: '
                        f'{a.title} — {a.description} (+{a.points_reward} pts)</div>',
                        unsafe_allow_html=True)

            # Response metrics
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Response",   response.response_code.value[:12])
            r2.metric("Verified",   f"{response.verified_count}/{pkg.total_signals}")
            r3.metric("Bonus Pts",  f"+{response.bonus_points}")
            r4.metric("Rep Δ",      f"{response.rep_delta:+d}")

            # Transmission stats
            st.markdown(
                f'<div class="hq-label">TX: {pkg.payload_mb:.2f} MB raw → '
                f'{pkg.compressed_mb:.2f} MB compressed → '
                f'{pkg.total_on_wire_mb:.2f} MB on-wire | '
                f'{pkg.transmission_time_s:.0f}s estimated uplink time</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="hq-label">— ADMIN ACTIONS —</div>', unsafe_allow_html=True)
        if st.button("⬡ ERASE ACTIVE DRIVE (VOID PROTOCOL)"):
            if dm and dm.active_drive:
                erased_id = dm.active_drive_id
                dm.erase_drive(erased_id)
                stats.drives_erased += 1
                ach.check_and_unlock(stats, archive, rep_value,
                                      recent_event="VOID_ERASED")
                st.warning(f"Drive {erased_id} erased. VOID PROTOCOL COMPLETE.")
            else:
                st.error("No active drive to erase.")

        if st.button("⬡ MARK 03:33 EVENT"):
            ach.check_and_unlock(stats, archive, rep_value, recent_event="03:33")
            st.info("03:33 event logged.")

        if st.button("⬡ ADVANCE DAY"):
            lb.record_day(stats.day_number, stats.points_by_day.get(stats.day_number, 0),
                           stats.total_signals, stats.anomaly_count, rep_value, stats.highest_wow)
            stats.day_number += 1
            st.session_state.day = stats.day_number
            ach.check_and_unlock(stats, archive, rep_value)
            st.success(f"Advanced to Day {stats.day_number}")

    with col_main:
        tabs = st.tabs(["[ ARCHIVE ]",
                        "[ SCORING ANALYTICS ]",
                        "[ SUBMISSIONS ]",
                        "[ ACHIEVEMENTS ]",
                        "[ SESSION REPORT ]",
                        "[ LEADERBOARD ]"])

        with tabs[0]:
            archive_df = archive.to_dataframe()
            if not archive_df.empty:
                # WoW distribution
                fig_wow = viz.plot_wow_distribution(archive)
                st.pyplot(fig_wow, use_container_width=True)
                plt.close(fig_wow)

                # Class distribution
                fig_cls = viz.plot_class_distribution(archive)
                st.pyplot(fig_cls, use_container_width=True)
                plt.close(fig_cls)

                # Full archive table
                with st.expander("[ FULL ARCHIVE TABLE ]"):
                    st.dataframe(archive_df, use_container_width=True, hide_index=True)

                # Stats by class
                st.markdown('<div class="hq-label">— STATS BY CLASS —</div>', unsafe_allow_html=True)
                cls_df = archive.stats_by_class()
                if not cls_df.empty:
                    st.dataframe(cls_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="hq-label">NO SIGNALS ARCHIVED YET. INGEST FROM SIGNAL LOG.</div>',
                            unsafe_allow_html=True)

        with tabs[1]:
            if archive.get_all():
                fig_pts = viz.plot_points_timeline(stats, archive)
                st.pyplot(fig_pts, use_container_width=True)
                plt.close(fig_pts)

                # Scoring breakdown table
                scoring_eng = ScoringEngine()
                sc_rows = []
                for sig_class, base in SIGNAL_BASE_POINTS.items():
                    for wow_lo, wow_hi, w_mult in WOW_MULTIPLIERS:
                        pts = int(base * w_mult * 1.0 * 1.0 * 1.0 * 1.0)
                        sc_rows.append({
                            "Class":    sig_class,
                            "WoW Range": f"{wow_lo}–{wow_hi}",
                            "Base":     base,
                            "WoW×":    w_mult,
                            "Nominal Pts": pts,
                        })
                st.markdown('<div class="hq-label">— SCORING MATRIX —</div>', unsafe_allow_html=True)
                sc_df = pd.DataFrame(sc_rows)
                st.dataframe(sc_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="hq-label">INGEST SIGNALS TO SEE ANALYTICS.</div>',
                            unsafe_allow_html=True)

        with tabs[2]:
            fig_sub = viz.plot_submission_history(pkgs)
            st.pyplot(fig_sub, use_container_width=True)
            plt.close(fig_sub)

            if pkgs:
                pkg_rows = [{
                    "Package ID": p.package_id,
                    "Drive":      p.drive_id,
                    "Signals":    p.total_signals,
                    "Points":     p.total_points,
                    "ARIRAL":     p.ariral_signals,
                    "WoW Max":    round(p.wow_max, 2),
                    "Size (MB)":  round(p.total_on_wire_mb, 3),
                    "TX Time(s)": round(p.transmission_time_s, 0),
                    "Status":     p.status.value,
                    "HQ Code":    p.hq_response or "—",
                } for p in pkgs]
                st.dataframe(pd.DataFrame(pkg_rows),
                             use_container_width=True, hide_index=True)

        with tabs[3]:
            fig_ach = viz.plot_achievement_grid(ach)
            st.pyplot(fig_ach, use_container_width=True)
            plt.close(fig_ach)

            st.dataframe(ach.to_dataframe(), use_container_width=True, hide_index=True)
            st.markdown(
                f'<div class="hq-label">UNLOCKED: {ach.unlocked_count()}/{len(ach._achievements)} | '
                f'ACHIEVEMENT POINTS: {ach.total_achievement_points()}</div>',
                unsafe_allow_html=True)

        with tabs[4]:
            rep_mgr = st.session_state.get("rep_manager", None)
            rep_val = int(getattr(getattr(rep_mgr, "rep", None), "current", 0))
            report_txt = rgen.generate_session_report(stats, archive, ach, pkgs, rep_val)
            st.markdown(f'<div class="hq-bulletin">{report_txt}</div>',
                        unsafe_allow_html=True)

        with tabs[5]:
            lb_df = lb.to_dataframe()
            if not lb_df.empty:
                st.dataframe(lb_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="hq-label">SUBMIT DRIVES TO POPULATE LEADERBOARD.</div>',
                            unsafe_allow_html=True)

            # Global session summary
            st.markdown('<div class="hq-label">— SESSION SUMMARY —</div>', unsafe_allow_html=True)
            ss1, ss2, ss3, ss4 = st.columns(4)
            ss1.metric("Total Points",  f"{stats.total_points:,}")
            ss2.metric("Total Signals", stats.total_signals)
            ss3.metric("Highest WoW",   f"{stats.highest_wow:.2f}")
            ss4.metric("Highest SNR",   f"{stats.highest_snr:.1f} dB")


if __name__ == "__main__":
    hq_reporter_page()
