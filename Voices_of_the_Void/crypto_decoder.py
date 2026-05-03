"""
crypto_decoder.py — Signal Cryptography & Message Decoding Engine
Voices of the Void | Alpen Signal Observatorium — Dunkeltaler Forest
Cipher Intelligence & Linguistic Analysis System v1.8.4

Handles: Binary stream extraction from complex baseband signals, Morse code
         detection and decoding, ASCII/UTF-8 deframing, prime-sequence
         encoding/decoding (ARIRAL communication standard), Fibonacci word
         message extraction, Vigenère cipher cryptanalysis via Index of
         Coincidence, Caesar/ROT-N brute force, Huffman frequency
         reconstruction, XOR key extraction via known-plaintext attack,
         base64/base32/base58 detection and decoding, Shannon entropy
         decomposition, Kolmogorov-Smirnov linguistic plausibility test,
         n-gram frequency distribution, ARIRAL message template matching,
         spectral steganography (LSB extraction from waterfall), run-length
         encoding analysis, message integrity (HMAC-SHA256), decoded message
         archive, candidate ranking, transmission timeline.

"If something is trying to speak, it is your obligation to listen."
                               — Dr. Kel, observing log day 41
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import itertools
import math
import re
import string
import struct
import time
import uuid
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.fft as sp_fft
import scipy.stats as sp_stats
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MORSE_TABLE: Dict[str, str] = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6",
    "--...": "7", "---..": "8", "----.": "9", ".-.-.-": ".",
    "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&",
    "---...": ":", "-.-.-.": ";", "-...-": "=", ".-.-.": "+",
    "-....-": "-", "..--.-": "_", ".-..-.": '"', "...-..-": "$",
    ".--.-.": "@", "...---...": "SOS",
}
MORSE_REVERSE: Dict[str, str] = {v: k for k, v in MORSE_TABLE.items()}

# English letter frequency reference
ENGLISH_FREQ: Dict[str, float] = {
    "E": 0.1202, "T": 0.0910, "A": 0.0812, "O": 0.0768, "I": 0.0731,
    "N": 0.0695, "S": 0.0628, "R": 0.0602, "H": 0.0592, "D": 0.0432,
    "L": 0.0398, "U": 0.0288, "C": 0.0271, "M": 0.0261, "F": 0.0230,
    "Y": 0.0211, "W": 0.0209, "G": 0.0203, "P": 0.0182, "B": 0.0149,
    "V": 0.0111, "K": 0.0069, "X": 0.0017, "Q": 0.0011, "J": 0.0010,
    "Z": 0.0007,
}

# ARIRAL prime word dictionary (fictional but internally consistent)
ARIRAL_PRIMES: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                              41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
                              89, 97, 101, 103, 107, 109, 113]

ARIRAL_SYMBOL_MAP: Dict[int, str] = {
    2:  "∅ (void/null)",      3:  "◈ (signal)",
    5:  "◉ (observer)",       7:  "⊕ (contact)",
    11: "⊗ (warning)",       13: "△ (approach)",
    17: "▽ (retreat)",        19: "◇ (gift)",
    23: "⬡ (station)",        29: "⬢ (array)",
    31: "⊙ (star-system)",    37: "⊛ (danger)",
    41: "◎ (acknowledge)",    43: "⊜ (transmit)",
    47: "⊝ (receive)",        53: "⬟ (time)",
    59: "⬠ (frequency)",      61: "⬡ (entity)",
    67: "⊞ (end)",            71: "⊟ (begin)",
    73: "⊠ (unknown)",        79: "⊡ (home)",
    83: "⋈ (bridge)",         89: "⋉ (threshold)",
    97: "⋊ (beyond)",
}

KNOWN_ARIRAL_PHRASES: List[Tuple[List[int], str]] = [
    ([3, 5, 41],       "SIGNAL — OBSERVER — ACKNOWLEDGE"),
    ([5, 29, 7],       "OBSERVER — ARRAY — CONTACT"),
    ([7, 19, 5],       "CONTACT — GIFT — OBSERVER"),
    ([11, 13, 29],     "WARNING — APPROACH — ARRAY"),
    ([11, 17, 66],     "WARNING — RETREAT — [unknown]"),
    ([71, 3, 41, 67],  "BEGIN — SIGNAL — ACKNOWLEDGE — END"),
    ([5, 89, 97],      "OBSERVER — THRESHOLD — BEYOND"),
    ([2, 67],          "VOID — END"),
    ([37, 11, 17],     "DANGER — WARNING — RETREAT"),
    ([3, 61, 7, 41],   "SIGNAL — ENTITY — CONTACT — ACKNOWLEDGE"),
]

SECRET_KEY = b"dunkeltaler_void_obs_2024_CLASSIFIED"

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class EncodingType(Enum):
    BINARY_ASK      = "BINARY_ASK"      # Amplitude-shift keying
    MORSE           = "MORSE"           # International Morse
    ASCII_8BIT      = "ASCII_8BIT"      # Standard ASCII
    PRIME_SEQUENCE  = "PRIME_SEQUENCE"  # ARIRAL prime encoding
    FIBONACCI_WORD  = "FIBONACCI_WORD"  # Fibonacci substitution
    BASE64          = "BASE64"          # Base-64 encoding
    BASE32          = "BASE32"          # Base-32 encoding
    RUN_LENGTH      = "RUN_LENGTH"      # RLE compressed
    HUFFMAN_LIKE    = "HUFFMAN_LIKE"    # Variable-length codes
    XOR_CIPHER      = "XOR_CIPHER"      # XOR with repeating key
    VIGENERE        = "VIGENERE"        # Polyalphabetic substitution
    CAESAR          = "CAESAR"          # Simple shift cipher
    STEGANOGRAPHIC  = "STEGANOGRAPHIC"  # Hidden in spectral LSBs
    UNKNOWN         = "UNKNOWN"

class DecodingConfidence(Enum):
    DEFINITIVE  = "DEFINITIVE"   # > 90% confidence
    HIGH        = "HIGH"         # 70–90%
    MODERATE    = "MODERATE"     # 50–70%
    SPECULATIVE = "SPECULATIVE"  # < 50%
    FAILED      = "FAILED"

class MessageClass(Enum):
    NATURAL_LANGUAGE   = "NATURAL_LANGUAGE"
    STRUCTURED_DATA    = "STRUCTURED_DATA"
    ARIRAL_LANGUAGE    = "ARIRAL_LANGUAGE"
    MATHEMATICAL       = "MATHEMATICAL"
    NOISE              = "NOISE"
    VOID_TRANSMISSION  = "VOID_TRANSMISSION"
    UNKNOWN            = "UNKNOWN"

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BinaryStream:
    bits:            np.ndarray       # 0/1 array
    bit_rate_bps:    float
    extraction_snr:  float
    method:          str
    n_errors:        int = 0
    sync_word_found: bool = False

    @property
    def n_bits(self) -> int:
        return len(self.bits)

    @property
    def byte_array(self) -> bytes:
        n = (len(self.bits) // 8) * 8
        out = bytearray()
        for i in range(0, n, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(self.bits[i + j])
            out.append(byte)
        return bytes(out)

    def hex_dump(self, width: int = 16) -> str:
        ba = self.byte_array
        lines = []
        for i in range(0, len(ba), width):
            chunk = ba[i:i + width]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:04X}  {hex_part:<{width * 3}}  {asc_part}")
        return "\n".join(lines)


@dataclass
class DecodedMessage:
    uid:             str   = field(default_factory=lambda: str(uuid.uuid4())[:10].upper())
    timestamp:       float = field(default_factory=time.time)
    encoding_type:   EncodingType    = EncodingType.UNKNOWN
    message_class:   MessageClass    = MessageClass.UNKNOWN
    confidence:      DecodingConfidence = DecodingConfidence.FAILED
    confidence_score: float = 0.0    # 0–1
    raw_bits:        int   = 0
    raw_bytes:       int   = 0
    plaintext:       str   = ""
    ariral_symbols:  List[str] = field(default_factory=list)
    ariral_phrase:   str   = ""
    language_score:  float = 0.0     # Closeness to English/natural language
    entropy_bits:    float = 0.0     # Shannon entropy of plaintext
    cipher_key:      Optional[str]  = None
    key_length:      int   = 0
    checksum_valid:  bool  = False
    hmac_tag:        str   = ""
    notes:           str   = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "UID":       self.uid,
            "Time":      pd.Timestamp(self.timestamp, unit="s").strftime("%H:%M:%S"),
            "Encoding":  self.encoding_type.value,
            "Class":     self.message_class.value,
            "Conf":      self.confidence.value,
            "Conf%":     f"{self.confidence_score*100:.0f}%",
            "Length":    len(self.plaintext),
            "Entropy":   f"{self.entropy_bits:.3f}",
            "Lang%":     f"{self.language_score*100:.0f}%",
            "Key":       self.cipher_key or "—",
            "✓":         "✓" if self.checksum_valid else "—",
            "Preview":   (self.plaintext[:50] + "…") if len(self.plaintext) > 50 else self.plaintext,
            "ARIRAL":    self.ariral_phrase[:40] if self.ariral_phrase else "—",
        }


@dataclass
class CipherAnalysis:
    method:         str
    key:            str
    key_length:     int
    plaintext:      str
    ic_score:       float   # Index of Coincidence (1.73 for English)
    chi2_score:     float   # Chi-squared against English frequency
    confidence:     float   # 0–1


@dataclass
class FrequencyProfile:
    symbol_counts:  Dict[str, int]
    total_symbols:  int
    entropy:        float
    most_common:    List[Tuple[str, int]]
    ic:             float   # Index of Coincidence
    chi2_vs_english: float
    is_likely_natural: bool


# ─────────────────────────────────────────────────────────────────────────────
# BINARY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class BinaryExtractor:
    """
    Extracts binary bit streams from complex baseband signals.
    Implements: threshold detection (ASK), zero-crossing detection,
    clock recovery (Gardner), sync-word search.
    """

    SYNC_WORDS: List[bytes] = [
        b"\xAA\xAA\xAA\xD3\x91",   # common preamble
        b"\xFF\xFE",                 # BOM-like
        b"\xEB\x90\x90",            # x86 NOP sled indicator
        b"\x55\xAA",                # boot sector
        b"\x1A\xCF",                # ARIRAL preamble (fictional)
    ]

    def __init__(self, sample_rate: float = 2e6):
        self.sr = sample_rate

    def extract_ask(self, signal: np.ndarray,
                    bit_rate_bps: float = 1000.0,
                    threshold_quantile: float = 0.5
                    ) -> BinaryStream:
        """
        Amplitude-Shift Keying extraction.
        Envelope thresholded at median; clock-synchronised by Gardner algorithm.
        """
        env = np.abs(signal)
        # Matched filter: integrate over one bit period
        samp_per_bit = max(1, int(self.sr / bit_rate_bps))
        kernel = np.ones(samp_per_bit) / samp_per_bit
        smoothed = np.convolve(env, kernel, mode="same")

        threshold = float(np.quantile(smoothed, threshold_quantile))
        bits_raw = (smoothed > threshold).astype(np.uint8)

        # Downsample to one sample per bit using clock recovery
        bits = self._clock_recover(bits_raw, samp_per_bit)
        snr = float(np.abs(np.mean(smoothed[bits_raw == 1]) -
                            np.mean(smoothed[bits_raw == 0])) /
                    (np.std(smoothed) + 1e-10))

        n_errors = int(np.sum(np.abs(np.diff(bits_raw.astype(int))) > 0))
        sync_found = self._search_sync(bits)

        return BinaryStream(
            bits=bits,
            bit_rate_bps=bit_rate_bps,
            extraction_snr=snr,
            method="ASK_THRESHOLD",
            n_errors=n_errors,
            sync_word_found=sync_found,
        )

    def _clock_recover(self, bits_raw: np.ndarray,
                        samp_per_bit: int) -> np.ndarray:
        """Downsample continuous bit stream to one sample per bit."""
        n = len(bits_raw)
        out: List[int] = []
        # Find first rising edge for alignment
        edges = np.where(np.diff(bits_raw.astype(int)) != 0)[0]
        offset = int(edges[0]) if len(edges) > 0 else 0
        t = offset + samp_per_bit // 2
        while t < n:
            out.append(int(bits_raw[t]))
            t += samp_per_bit
        return np.array(out, dtype=np.uint8)

    def _search_sync(self, bits: np.ndarray) -> bool:
        """Search for known sync words in the bit stream."""
        ba = BinaryStream(bits, 0, 0, "").byte_array
        for sw in self.SYNC_WORDS:
            if sw in ba:
                return True
        return False

    def extract_zero_crossing(self, signal: np.ndarray,
                               bit_rate_bps: float = 1000.0) -> BinaryStream:
        """
        Phase-based extraction: detect bit transitions via zero crossings
        of the real part. Used for BPSK-like signals.
        """
        real = signal.real
        samp_per_bit = max(1, int(self.sr / bit_rate_bps))
        zc = np.where(np.diff(np.sign(real)))[0]
        bits: List[int] = []
        last_sign = int(np.sign(real[0])) if len(real) > 0 else 1
        for i in range(0, len(real) - samp_per_bit, samp_per_bit):
            seg = real[i:i + samp_per_bit]
            dominant = int(np.sign(np.mean(seg))) if len(seg) > 0 else last_sign
            bits.append(0 if dominant >= 0 else 1)
            last_sign = dominant
        b_arr = np.array(bits, dtype=np.uint8)
        snr = float(np.abs(np.mean(real[real > 0])) /
                    (np.std(real) + 1e-10)) if len(real) > 0 else 0.0
        return BinaryStream(
            bits=b_arr,
            bit_rate_bps=bit_rate_bps,
            extraction_snr=snr,
            method="ZERO_CROSSING_BPSK",
            sync_word_found=self._search_sync(b_arr),
        )

    def extract_spectral_lsb(self, power_db_matrix: np.ndarray,
                              threshold_db: float = 3.0) -> BinaryStream:
        """
        Spectral steganography: extract LSBs from waterfall power matrix.
        Each time-frequency cell's LSB contributes one bit.
        """
        n_freq, n_time = power_db_matrix.shape
        # Quantise to integer dB; take LSBs
        quantised = np.round(power_db_matrix).astype(np.int32)
        lsbs = (quantised & 1).flatten().astype(np.uint8)
        # Only extract where power is above noise floor
        noise_floor = float(np.percentile(power_db_matrix, 10))
        mask = (power_db_matrix > noise_floor + threshold_db).flatten()
        bits = lsbs[mask]
        return BinaryStream(
            bits=bits,
            bit_rate_bps=1.0,
            extraction_snr=threshold_db,
            method="SPECTRAL_LSB_STEGO",
            sync_word_found=self._search_sync(bits),
        )

    def detect_bit_rate(self, signal: np.ndarray,
                         rates_to_try: Optional[List[float]] = None) -> float:
        """
        Estimate bit rate by finding the inter-symbol interval from
        the autocorrelation of the squared envelope.
        """
        if rates_to_try is None:
            rates_to_try = [100, 300, 1200, 2400, 4800, 9600, 19200, 115200]
        env2 = np.abs(signal) ** 2
        env2 -= env2.mean()
        n = len(env2)
        ac = np.correlate(env2, env2, mode="full")[n - 1:]
        ac = ac / (ac[0] + 1e-20)
        max_lag = min(n // 2, int(self.sr / 50))  # at most sr/50 lags

        best_rate, best_score = 300.0, -1.0
        for rate in rates_to_try:
            samp = self.sr / rate
            if samp < 2 or samp > max_lag:
                continue
            idx = int(round(samp))
            if idx < len(ac):
                score = float(ac[idx])
                if score > best_score:
                    best_score, best_rate = score, rate
        return best_rate


# ─────────────────────────────────────────────────────────────────────────────
# MORSE CODE DECODER
# ─────────────────────────────────────────────────────────────────────────────

class MorseDecoder:
    """
    Full Morse code detection and decoding.
    Auto-detects dit/dah ratio from pulse duration histogram.
    Handles inter-letter and inter-word spacing.
    """

    def detect_morse(self, signal: np.ndarray,
                      sample_rate: float = 2e6) -> Tuple[bool, float]:
        """
        Detect if signal contains Morse-like pulse structure.
        Returns (is_morse, confidence).
        Uses pulse-duration histogram: Morse should have 2–3 distinct clusters.
        """
        env = np.abs(signal)
        threshold = float(np.percentile(env, 60))
        on_off = (env > threshold).astype(np.uint8)
        runs = []
        i = 0
        while i < len(on_off):
            val = on_off[i]
            j = i
            while j < len(on_off) and on_off[j] == val:
                j += 1
            runs.append((int(val), j - i))
            i = j

        on_durations  = [d for v, d in runs if v == 1]
        off_durations = [d for v, d in runs if v == 0]
        if len(on_durations) < 5:
            return False, 0.0

        # Expect 2 clusters in on_durations (dit and dah) → ratio ~1:3
        on_arr = np.array(on_durations, dtype=float)
        median = float(np.median(on_arr))
        short  = on_arr[on_arr < median * 1.8]
        long_  = on_arr[on_arr >= median * 1.8]
        if len(short) < 2 or len(long_) < 1:
            return False, 0.0
        ratio = float(np.median(long_) / (np.median(short) + 1e-10))
        # Classic Morse: dah = 3× dit
        ratio_score = float(np.exp(-0.5 * ((ratio - 3.0) / 1.0) ** 2))
        confidence  = float(np.clip(ratio_score * len(on_durations) / 30, 0, 1))
        return confidence > 0.25, confidence

    def decode(self, signal: np.ndarray,
               sample_rate: float = 2e6) -> Tuple[str, str, float]:
        """
        Decode Morse code from signal.
        Returns (decoded_text, morse_string, confidence).
        """
        env = np.abs(signal)
        threshold = float(np.percentile(env, 55))
        on_off = (env > threshold).astype(np.uint8)

        # Extract run-length encoded sequence
        runs: List[Tuple[int, int]] = []
        i = 0
        while i < len(on_off):
            val = on_off[i]
            j = i
            while j < len(on_off) and on_off[j] == val:
                j += 1
            runs.append((int(val), j - i))
            i = j

        on_durs  = np.array([d for v, d in runs if v == 1], dtype=float)
        off_durs = np.array([d for v, d in runs if v == 0], dtype=float)

        if len(on_durs) < 3:
            return "", "", 0.0

        dit_dur  = float(np.percentile(on_durs, 35))
        dah_dur  = float(np.percentile(on_durs, 75))
        dit_thresh = (dit_dur + dah_dur) / 2
        char_gap   = dit_dur * 3.5
        word_gap   = dit_dur * 7.0

        # Build morse string
        morse_chars: List[str] = []
        morse_string_parts: List[str] = []
        current_code = ""
        run_idx = 0

        for val, dur in runs:
            if val == 1:  # on → dit or dah
                if dur <= dit_thresh:
                    current_code += "."
                else:
                    current_code += "-"
            else:         # off → separator
                if dur >= word_gap:
                    if current_code:
                        letter = MORSE_TABLE.get(current_code, "?")
                        morse_chars.append(letter)
                        morse_string_parts.append(current_code)
                        current_code = ""
                    morse_chars.append(" ")
                    morse_string_parts.append("/")
                elif dur >= char_gap:
                    if current_code:
                        letter = MORSE_TABLE.get(current_code, "?")
                        morse_chars.append(letter)
                        morse_string_parts.append(current_code)
                        current_code = ""

        if current_code:
            letter = MORSE_TABLE.get(current_code, "?")
            morse_chars.append(letter)
            morse_string_parts.append(current_code)

        decoded = "".join(morse_chars).strip()
        morse_str = " ".join(morse_string_parts)
        n_total = len(morse_chars)
        n_valid = sum(1 for c in morse_chars if c != "?")
        conf = float(n_valid / max(n_total, 1))
        return decoded, morse_str, conf


# ─────────────────────────────────────────────────────────────────────────────
# ASCII / UTF-8 DEFRAMER
# ─────────────────────────────────────────────────────────────────────────────

class ASCIIDeframer:
    """
    Extracts human-readable text from binary streams.
    Handles 7-bit ASCII, 8-bit extended ASCII, and UTF-8.
    Performs sync-word searching, parity checking, and printability scoring.
    """

    def decode(self, stream: BinaryStream) -> Tuple[str, float]:
        """
        Attempt ASCII decoding. Returns (text, printability_score).
        """
        ba = stream.byte_array
        if not ba:
            return "", 0.0

        # Try UTF-8 first
        for encoding in ("utf-8", "latin-1", "ascii"):
            try:
                text = ba.decode(encoding, errors="replace")
                score = self._printability_score(text)
                if score > 0.6:
                    return text, score
            except Exception:
                continue

        # Raw ASCII (7-bit only)
        text_7 = "".join(chr(b & 0x7F) for b in ba)
        score  = self._printability_score(text_7)
        return text_7, score

    @staticmethod
    def _printability_score(text: str) -> float:
        """Fraction of printable characters."""
        if not text:
            return 0.0
        printable = set(string.printable)
        return sum(1 for c in text if c in printable) / len(text)

    def extract_strings(self, stream: BinaryStream,
                         min_length: int = 4) -> List[str]:
        """Find all printable string runs of ≥ min_length chars."""
        ba = stream.byte_array
        current = ""
        results = []
        for byte in ba:
            c = chr(byte)
            if c in string.printable and c not in "\n\r\t":
                current += c
            else:
                if len(current) >= min_length:
                    results.append(current)
                current = ""
        if len(current) >= min_length:
            results.append(current)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# PRIME SEQUENCE DECODER (ARIRAL STANDARD)
# ─────────────────────────────────────────────────────────────────────────────

class PrimeSequenceDecoder:
    """
    Decodes prime-number-based encoding used by ARIRAL class signals.
    The encoding embeds sequences of prime numbers as inter-peak spacings
    or as modulation frequencies. Each prime maps to an ARIRAL symbol.
    """

    def __init__(self):
        self._prime_set = set(ARIRAL_PRIMES)
        self._prime_sorted = sorted(ARIRAL_PRIMES)

    def extract_prime_sequence(self, signal: np.ndarray,
                                sample_rate: float = 2e6,
                                n_fft: int = 2048
                                ) -> Tuple[List[int], float]:
        """
        Extract sequence of primes from spectral peak locations.
        Peak bin indices are tested for primeness and their sequence decoded.
        Returns (prime_sequence, confidence).
        """
        pwr = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
        freqs_bins = np.arange(len(pwr))
        peaks, props = sp_signal.find_peaks(
            pwr,
            height=np.mean(pwr) + 2 * np.std(pwr),
            distance=3,
            prominence=np.std(pwr),
        )
        if len(peaks) < 3:
            return [], 0.0

        # Check peak bin indices for primeness
        prime_peaks = [int(p) for p in peaks if self._is_prime(int(p))]
        non_prime   = [int(p) for p in peaks if not self._is_prime(int(p))]

        if len(peaks) == 0:
            return [], 0.0
        prime_fraction = len(prime_peaks) / len(peaks)

        # Map to nearest ARIRAL prime
        sequence = []
        for pp in sorted(prime_peaks):
            nearest = min(self._prime_sorted, key=lambda x: abs(x - pp))
            if abs(nearest - pp) < pp * 0.2:
                sequence.append(nearest)

        confidence = float(min(1.0, prime_fraction * 2.0 * len(sequence) / max(len(peaks), 1)))
        return sequence, confidence

    def decode_sequence(self, primes: List[int]) -> Tuple[List[str], str, float]:
        """
        Map prime sequence to ARIRAL symbols.
        Returns (symbol_list, phrase_match, phrase_confidence).
        """
        if not primes:
            return [], "", 0.0

        symbols = [ARIRAL_SYMBOL_MAP.get(p, f"[p={p}]") for p in primes]

        # Try to match known ARIRAL phrases
        best_phrase, best_conf = "", 0.0
        for phrase_primes, phrase_text in KNOWN_ARIRAL_PHRASES:
            if len(phrase_primes) == 0:
                continue
            # Sliding window match
            win = len(phrase_primes)
            for start in range(max(1, len(primes) - win + 1)):
                window = primes[start:start + win]
                matches = sum(1 for a, b in zip(window, phrase_primes) if a == b)
                conf = matches / len(phrase_primes)
                if conf > best_conf:
                    best_conf = conf
                    best_phrase = phrase_text

        return symbols, best_phrase, float(best_conf)

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def fibonacci_word_decode(self, stream: BinaryStream) -> Tuple[str, float]:
        """
        Decode a Fibonacci word substitution: 0→'A', 1→'B'.
        The Fibonacci word (1,10,101,10110,...) is the infinite word
        generated by σ(1)=10, σ(0)=1.
        Confidence = fraction of bits matching Fibonacci-word statistics.
        """
        bits = stream.bits
        if len(bits) < 8:
            return "", 0.0

        # Fibonacci word statistics: density of '1' → φ-1 ≈ 0.618
        phi_inv = (math.sqrt(5) - 1) / 2
        observed_density = float(np.mean(bits))
        density_score = float(np.exp(-10 * (observed_density - phi_inv) ** 2))

        # Decode as substitution: 1→B, 0→A
        text = "".join("B" if b else "A" for b in bits)
        confidence = float(density_score)
        return text[:200], confidence


# ─────────────────────────────────────────────────────────────────────────────
# CIPHER ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class CipherAnalyser:
    """
    Cryptanalysis suite: frequency analysis, Index of Coincidence,
    Kasiski/Friedman Vigenère key length estimation, key recovery,
    Caesar brute force, XOR key extraction.
    """

    # ── Frequency analysis ────────────────────────────────────────────────────

    @staticmethod
    def frequency_profile(text: str) -> FrequencyProfile:
        """Compute letter frequency statistics for a text string."""
        text_up = text.upper()
        letters = [c for c in text_up if c.isalpha()]
        n = len(letters)
        if n == 0:
            return FrequencyProfile({}, 0, 0.0, [], 0.0, 0.0, False)
        counts = Counter(letters)
        total  = sum(counts.values())

        # Index of Coincidence
        ic = sum(v * (v - 1) for v in counts.values()) / (n * (n - 1) + 1e-10)

        # Chi-squared against English
        chi2 = 0.0
        for letter in string.ascii_uppercase:
            obs = counts.get(letter, 0) / n
            exp = ENGLISH_FREQ.get(letter, 0.001)
            chi2 += (obs - exp) ** 2 / (exp + 1e-10)

        # Shannon entropy (symbol level)
        probs = np.array([v / total for v in counts.values()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-30)))

        return FrequencyProfile(
            symbol_counts=dict(counts),
            total_symbols=total,
            entropy=entropy,
            most_common=counts.most_common(10),
            ic=float(ic),
            chi2_vs_english=float(chi2),
            is_likely_natural=bool(ic > 0.060 and chi2 < 0.05),
        )

    # ── Index of Coincidence ────────────────────────────────────────────────────

    @staticmethod
    def index_of_coincidence(text: str) -> float:
        """IC ≈ 0.0385 (uniform), 0.0667 (English), 0.0286 (random)."""
        letters = [c for c in text.upper() if c.isalpha()]
        n = len(letters)
        if n < 2:
            return 0.0
        counts = Counter(letters)
        return float(sum(v * (v - 1) for v in counts.values()) / (n * (n - 1)))

    # ── Caesar brute force ────────────────────────────────────────────────────

    def break_caesar(self, ciphertext: str) -> CipherAnalysis:
        """Brute-force all 26 shifts; rank by chi-squared against English."""
        letters = [c for c in ciphertext.upper() if c.isalpha()]
        n = len(letters)
        best: Optional[CipherAnalysis] = None

        for shift in range(26):
            plaintext_letters = [chr((ord(c) - 65 - shift) % 26 + 65) for c in letters]
            counts = Counter(plaintext_letters)
            chi2 = 0.0
            for letter in string.ascii_uppercase:
                obs = counts.get(letter, 0) / max(n, 1)
                exp = ENGLISH_FREQ.get(letter, 0.001)
                chi2 += (obs - exp) ** 2 / (exp + 1e-10)
            ic  = self.index_of_coincidence("".join(plaintext_letters))
            conf = float(np.exp(-chi2 * 2) * (ic / 0.0667))

            # Reconstruct full plaintext
            plain = self._apply_caesar("".join(plaintext_letters), 0,
                                        ciphertext, shift)
            if best is None or chi2 < best.chi2_score:
                best = CipherAnalysis(
                    method=f"CAESAR_ROT{shift}",
                    key=str(shift),
                    key_length=1,
                    plaintext=plain[:500],
                    ic_score=ic,
                    chi2_score=chi2,
                    confidence=float(np.clip(conf, 0, 1)),
                )
        return best or CipherAnalysis("CAESAR", "0", 1, ciphertext, 0.0, 999.0, 0.0)

    @staticmethod
    def _apply_caesar(letters: str, shift: int,
                       original: str, actual_shift: int) -> str:
        """Reconstruct plaintext preserving original punctuation/spaces."""
        letter_gen = iter(letters)
        out = []
        for c in original:
            if c.isalpha():
                try:
                    out.append(next(letter_gen))
                except StopIteration:
                    out.append(c)
            else:
                out.append(c)
        return "".join(out)

    # ── Vigenère key length (Friedman test) ────────────────────────────────────

    def friedman_key_length(self, ciphertext: str, max_len: int = 20
                             ) -> List[Tuple[int, float]]:
        """
        Estimate Vigenère key length via Index of Coincidence method.
        For each trial key length k, compute average IC of k subsequences.
        Best key length → IC closest to 0.0667 (English).
        """
        text = [c for c in ciphertext.upper() if c.isalpha()]
        n = len(text)
        results = []
        for k in range(2, min(max_len + 1, n // 3)):
            ics = []
            for i in range(k):
                subseq = "".join(text[j] for j in range(i, n, k))
                ics.append(self.index_of_coincidence(subseq))
            avg_ic = float(np.mean(ics))
            dist = abs(avg_ic - 0.0667)
            results.append((k, avg_ic))
        results.sort(key=lambda x: abs(x[1] - 0.0667))
        return results[:5]

    # ── Vigenère key recovery ────────────────────────────────────────────────

    def break_vigenere(self, ciphertext: str,
                        max_key_len: int = 15) -> CipherAnalysis:
        """
        Full Vigenère cryptanalysis:
        1. Friedman key length estimation
        2. Per-column Caesar attack
        Returns best decryption.
        """
        text = [c for c in ciphertext.upper() if c.isalpha()]
        n = len(text)
        if n < 20:
            return CipherAnalysis("VIGENERE", "?", 0, ciphertext, 0.0, 999.0, 0.0)

        key_lens = self.friedman_key_length(ciphertext, max_len=max_key_len)
        best: Optional[CipherAnalysis] = None

        for k, avg_ic in key_lens[:3]:
            key_chars = []
            for i in range(k):
                col = "".join(text[j] for j in range(i, n, k))
                # Frequency shift: find shift that minimises chi2
                col_counts = Counter(col)
                best_shift, best_chi2 = 0, float("inf")
                for shift in range(26):
                    chi2 = 0.0
                    for letter in string.ascii_uppercase:
                        obs = col_counts.get(chr((ord(letter) - 65 + shift) % 26 + 65), 0) / max(len(col), 1)
                        exp = ENGLISH_FREQ.get(letter, 0.001)
                        chi2 += (obs - exp) ** 2 / (exp + 1e-10)
                    if chi2 < best_chi2:
                        best_chi2, best_shift = chi2, shift
                key_chars.append(chr(best_shift + 65))

            key = "".join(key_chars)
            # Decrypt
            out, key_idx = [], 0
            for c in ciphertext.upper():
                if c.isalpha():
                    shift = ord(key[key_idx % k]) - 65
                    decrypted_c = chr((ord(c) - 65 - shift) % 26 + 65)
                    out.append(decrypted_c)
                    key_idx += 1
                else:
                    out.append(c)
            plaintext = "".join(out)
            ic   = self.index_of_coincidence(plaintext)
            prof = self.frequency_profile(plaintext)
            conf = float(np.clip((ic / 0.0667) * np.exp(-prof.chi2_vs_english), 0, 1))

            if best is None or (ic > (best.ic_score if best else 0)):
                best = CipherAnalysis(
                    method="VIGENERE",
                    key=key,
                    key_length=k,
                    plaintext=plaintext[:500],
                    ic_score=ic,
                    chi2_score=prof.chi2_vs_english,
                    confidence=conf,
                )
        return best or CipherAnalysis("VIGENERE", "?", 0, ciphertext, 0.0, 999.0, 0.0)

    # ── XOR key extraction ────────────────────────────────────────────────────

    def break_xor(self, ciphertext_bytes: bytes,
                   max_key_len: int = 32) -> CipherAnalysis:
        """
        XOR cipher cryptanalysis via Hamming distance key length estimation
        and per-byte frequency analysis.
        """
        n = len(ciphertext_bytes)
        if n < 8:
            return CipherAnalysis("XOR", "?", 0, "", 0.0, 999.0, 0.0)

        # Key length estimation: minimum normalised Hamming distance
        best_kl, best_hdist = 1, float("inf")
        for kl in range(2, min(max_key_len + 1, n // 4)):
            blocks = [ciphertext_bytes[i:i + kl]
                      for i in range(0, min(n, kl * 8), kl)]
            if len(blocks) < 2:
                continue
            dists = []
            for a, b in zip(blocks[:-1], blocks[1:]):
                d = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
                dists.append(d / kl)
            avg_d = float(np.mean(dists))
            if avg_d < best_hdist:
                best_hdist, best_kl = avg_d, kl

        # Recover key byte by byte
        key_bytes = []
        for i in range(best_kl):
            col = bytes([ciphertext_bytes[j]
                          for j in range(i, n, best_kl)])
            best_byte, best_score = 0, -float("inf")
            for candidate in range(256):
                decrypted = bytes([b ^ candidate for b in col])
                printable = sum(1 for b in decrypted if 32 <= b < 127)
                score = printable / max(len(decrypted), 1)
                if score > best_score:
                    best_score, best_byte = score, candidate
            key_bytes.append(best_byte)

        key = bytes(key_bytes)
        plaintext_bytes = bytes([b ^ key[i % len(key)]
                                   for i, b in enumerate(ciphertext_bytes)])
        try:
            plaintext = plaintext_bytes.decode("latin-1")
        except Exception:
            plaintext = plaintext_bytes.hex()

        printable_score = sum(1 for c in plaintext if c in string.printable) / max(len(plaintext), 1)
        return CipherAnalysis(
            method="XOR",
            key=key.hex(),
            key_length=best_kl,
            plaintext=plaintext[:500],
            ic_score=self.index_of_coincidence(plaintext),
            chi2_score=0.0,
            confidence=float(printable_score),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BASE ENCODING DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class BaseEncodingDetector:
    """
    Detects and decodes base64, base32, base16 (hex) encodings.
    Also handles URL-safe variants and padded/unpadded forms.
    """

    BASE64_CHARS  = set(string.ascii_letters + string.digits + "+/=")
    BASE64U_CHARS = set(string.ascii_letters + string.digits + "-_=")
    BASE32_CHARS  = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=")
    BASE16_CHARS  = set(string.hexdigits)

    def detect_and_decode(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Try all base encodings. Returns list of (encoding_name, decoded_text, confidence).
        """
        results = []
        cleaned = text.strip().replace("\n", "").replace(" ", "")

        # Base16
        if all(c in self.BASE16_CHARS for c in cleaned) and len(cleaned) % 2 == 0:
            try:
                decoded = bytes.fromhex(cleaned).decode("latin-1")
                score = self._printability(decoded)
                results.append(("BASE16", decoded[:200], score))
            except Exception:
                pass

        # Base32
        cleaned32 = cleaned.upper()
        if all(c in self.BASE32_CHARS for c in cleaned32):
            try:
                pad_len = (8 - len(cleaned32) % 8) % 8
                decoded = base64.b32decode(cleaned32 + "=" * pad_len).decode("latin-1")
                score = self._printability(decoded)
                if score > 0.5:
                    results.append(("BASE32", decoded[:200], score))
            except Exception:
                pass

        # Base64 standard
        if all(c in self.BASE64_CHARS for c in cleaned):
            try:
                pad_len = (4 - len(cleaned) % 4) % 4
                decoded = base64.b64decode(cleaned + "=" * pad_len).decode("latin-1")
                score = self._printability(decoded)
                if score > 0.4:
                    results.append(("BASE64", decoded[:200], score))
            except Exception:
                pass

        # Base64 URL-safe
        if all(c in self.BASE64U_CHARS for c in cleaned):
            try:
                pad_len = (4 - len(cleaned) % 4) % 4
                decoded = base64.urlsafe_b64decode(cleaned + "=" * pad_len).decode("latin-1")
                score = self._printability(decoded)
                if score > 0.4:
                    results.append(("BASE64_URL", decoded[:200], score))
            except Exception:
                pass

        return sorted(results, key=lambda x: -x[2])

    @staticmethod
    def _printability(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c in string.printable) / len(text)


# ─────────────────────────────────────────────────────────────────────────────
# LINGUISTIC ANALYSER — Language plausibility scoring
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticAnalyser:
    """
    Determines whether decoded text is:
    - Natural language (English)
    - Structured data (JSON, CSV, binary protocol)
    - ARIRAL language (prime-sequence base)
    - Mathematical sequence
    - Noise (random)
    Using: IC, chi², bigram frequency, entropy, KS-test.
    """

    # Common English bigrams (top-20)
    ENGLISH_BIGRAMS = {
        "TH": 0.0356, "HE": 0.0307, "IN": 0.0243, "ER": 0.0205, "AN": 0.0199,
        "RE": 0.0185, "ON": 0.0176, "AT": 0.0149, "EN": 0.0145, "ND": 0.0135,
        "TI": 0.0134, "ES": 0.0134, "OR": 0.0128, "TE": 0.0120, "OF": 0.0117,
        "ED": 0.0117, "IS": 0.0113, "IT": 0.0112, "AL": 0.0109, "AR": 0.0107,
    }

    def score_english(self, text: str) -> float:
        """
        Score 0–1 for how English-like the text is.
        Combines: IC score, chi2, bigram frequency, printability.
        """
        if not text:
            return 0.0
        analyser = CipherAnalyser()
        prof = analyser.frequency_profile(text)

        # IC score (English IC ≈ 0.0667)
        ic_score = float(np.clip(1.0 - abs(prof.ic - 0.0667) / 0.0667, 0, 1))

        # Chi2 score
        chi2_score = float(np.exp(-prof.chi2_vs_english * 5))

        # Bigram score
        text_up = text.upper()
        bigrams_obs = Counter(text_up[i:i+2] for i in range(len(text_up)-1)
                              if text_up[i:i+2].isalpha())
        total_bg = sum(bigrams_obs.values())
        bg_score = 0.0
        if total_bg > 0:
            for bg, exp_freq in self.ENGLISH_BIGRAMS.items():
                obs_freq = bigrams_obs.get(bg, 0) / total_bg
                bg_score += min(obs_freq, exp_freq) / exp_freq
            bg_score /= len(self.ENGLISH_BIGRAMS)

        # Printability
        printable = sum(1 for c in text if c in string.printable)
        print_score = printable / max(len(text), 1)

        # Word-like tokens
        words = re.findall(r"[a-zA-Z]{3,}", text)
        word_score = float(np.clip(len(words) / max(len(text) / 5, 1), 0, 1))

        return float(0.25 * ic_score + 0.25 * chi2_score +
                     0.20 * bg_score + 0.15 * print_score + 0.15 * word_score)

    def classify_message(self, text: str,
                          ariral_conf: float = 0.0) -> MessageClass:
        """Route decoded text to a MessageClass."""
        if not text:
            return MessageClass.NOISE

        eng_score = self.score_english(text)

        # JSON / CSV / structured data
        try:
            import json as _json
            _json.loads(text)
            return MessageClass.STRUCTURED_DATA
        except Exception:
            pass
        if text.count(",") > 3 and "\n" in text:
            return MessageClass.STRUCTURED_DATA

        # Mathematical sequence (digits, operators)
        alnum = [c for c in text if c.isalnum()]
        digit_frac = sum(1 for c in alnum if c.isdigit()) / max(len(alnum), 1)
        if digit_frac > 0.7:
            return MessageClass.MATHEMATICAL

        if ariral_conf > 0.5:
            return MessageClass.ARIRAL_LANGUAGE

        if eng_score > 0.55:
            return MessageClass.NATURAL_LANGUAGE

        # Entropy-based: very low entropy = structured, very high = noise
        if len(text) > 10:
            prob = np.array([v / len(text) for v in Counter(text).values()])
            h = float(-np.sum(prob * np.log2(prob + 1e-30)))
            if h < 1.5:
                return MessageClass.STRUCTURED_DATA
            if h > 7.0:
                return MessageClass.NOISE

        return MessageClass.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# RUN-LENGTH ENCODING DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class RLEAnalyser:
    """
    Detects and decodes Run-Length Encoding.
    Also computes compression ratio as a complexity indicator.
    """

    def encode_rle(self, data: bytes) -> bytes:
        """RLE encode: (count, byte) pairs."""
        if not data:
            return b""
        out = bytearray()
        i = 0
        while i < len(data):
            val = data[i]
            count = 1
            while i + count < len(data) and data[i + count] == val and count < 255:
                count += 1
            out.extend([count, val])
            i += count
        return bytes(out)

    def decode_rle(self, data: bytes) -> bytes:
        """Attempt RLE decode: interpret as (count, byte) pairs."""
        if len(data) < 2 or len(data) % 2 != 0:
            return b""
        out = bytearray()
        for i in range(0, len(data) - 1, 2):
            count = data[i]
            val   = data[i + 1]
            if count == 0:
                return b""
            out.extend([val] * count)
        return bytes(out)

    def compression_ratio(self, data: bytes) -> float:
        """Ratio of RLE output to input size. < 1 means compressible."""
        if not data:
            return 1.0
        encoded = self.encode_rle(data)
        return len(encoded) / len(data)

    def is_likely_rle(self, data: bytes) -> Tuple[bool, float]:
        """Heuristic: if compression ratio < 0.7, data may be RLE-compressed."""
        cr = self.compression_ratio(data)
        confidence = float(np.clip(1.0 - cr, 0, 1))
        return cr < 0.7, confidence


# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY DECOMPOSER
# ─────────────────────────────────────────────────────────────────────────────

class EntropyDecomposer:
    """
    Multi-scale Shannon entropy analysis.
    Detects: uniform noise, low-entropy structure, partial encryption,
    hierarchical encoding layers.
    """

    def compute(self, data: bytes, block_size: int = 256) -> Dict[str, Any]:
        """Full entropy decomposition."""
        if not data:
            return {}

        # Byte-level entropy
        counts = Counter(data)
        probs  = np.array([v / len(data) for v in counts.values()])
        h_byte = float(-np.sum(probs * np.log2(probs + 1e-30)))

        # Block-level entropy variance (detect heterogeneous encoding)
        n_blocks = len(data) // block_size
        block_entropies = []
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            bc = Counter(block)
            bp = np.array([v / len(block) for v in bc.values()])
            bh = float(-np.sum(bp * np.log2(bp + 1e-30)))
            block_entropies.append(bh)

        # Bit-level entropy
        all_bits = []
        for byte in data[:min(len(data), 4096)]:
            for bit_pos in range(8):
                all_bits.append((byte >> bit_pos) & 1)
        bit_density = float(np.mean(all_bits))
        h_bit = float(-bit_density * math.log2(bit_density + 1e-30)
                      - (1 - bit_density) * math.log2(1 - bit_density + 1e-30))

        # Consecutive byte correlation
        if len(data) > 1:
            arr = np.frombuffer(data[:min(len(data), 4096)], dtype=np.uint8).astype(float)
            corr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        else:
            corr = 0.0

        # Classification
        if h_byte > 7.8:
            classification = "ENCRYPTED_OR_RANDOM"
        elif h_byte > 6.0:
            classification = "COMPRESSED_OR_ENCODED"
        elif h_byte > 3.0:
            classification = "STRUCTURED_DATA"
        elif h_byte > 1.0:
            classification = "REPETITIVE_STRUCTURE"
        else:
            classification = "HIGHLY_REPETITIVE"

        return {
            "byte_entropy":           round(h_byte, 4),
            "bit_entropy":            round(h_bit, 4),
            "bit_density":            round(bit_density, 4),
            "byte_correlation":       round(corr, 4),
            "n_unique_bytes":         len(counts),
            "block_entropy_mean":     round(float(np.mean(block_entropies)), 4) if block_entropies else 0,
            "block_entropy_std":      round(float(np.std(block_entropies)), 4) if block_entropies else 0,
            "classification":         classification,
            "likely_plaintext":       bool(2.5 < h_byte < 5.5),
            "likely_cipher":          bool(h_byte > 7.5),
        }

    def entropy_timeline(self, data: bytes, window: int = 64) -> np.ndarray:
        """Compute rolling entropy across data (byte units)."""
        n = len(data)
        entropies = []
        for i in range(0, n - window, window // 2):
            block = data[i:i + window]
            bc = Counter(block)
            bp = np.array([v / len(block) for v in bc.values()])
            bh = float(-np.sum(bp * np.log2(bp + 1e-30)))
            entropies.append(bh)
        return np.array(entropies)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DECODER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class MasterDecoder:
    """
    Orchestrates all decoding attempts on a given signal or binary stream.
    Returns ranked list of DecodedMessage candidates.
    """

    def __init__(self, sample_rate: float = 2e6):
        self.sr           = sample_rate
        self.extractor    = BinaryExtractor(sample_rate)
        self.morse        = MorseDecoder()
        self.ascii_frame  = ASCIIDeframer()
        self.prime_dec    = PrimeSequenceDecoder()
        self.cipher       = CipherAnalyser()
        self.base_enc     = BaseEncodingDetector()
        self.rle          = RLEAnalyser()
        self.entropy      = EntropyDecomposer()
        self.ling         = LinguisticAnalyser()

    def decode_signal(self, signal: np.ndarray,
                       waterfall: Optional[np.ndarray] = None
                       ) -> List[DecodedMessage]:
        """
        Full decoding pipeline. Attempts all methods; returns all candidates
        sorted by confidence descending.
        """
        results: List[DecodedMessage] = []

        # ── 1. Bit-rate estimation ─────────────────────────────────────────
        bit_rate = self.extractor.detect_bit_rate(signal)

        # ── 2. Binary extraction — ASK and BPSK ───────────────────────────
        stream_ask  = self.extractor.extract_ask(signal, bit_rate)
        stream_bpsk = self.extractor.extract_zero_crossing(signal, bit_rate)

        for stream, method_label in [(stream_ask, "ASK"), (stream_bpsk, "BPSK")]:
            if stream.n_bits < 8:
                continue

            # ── 2a. ASCII decode ───────────────────────────────────────────
            txt, print_score = self.ascii_frame.decode(stream)
            if print_score > 0.5 and len(txt) > 3:
                lang_s = self.ling.score_english(txt)
                ent    = self.entropy.compute(stream.byte_array)
                h      = ent.get("byte_entropy", 0.0)
                msg_cls = self.ling.classify_message(txt)
                conf_s = float(print_score * 0.5 + lang_s * 0.5)
                results.append(DecodedMessage(
                    encoding_type=EncodingType.ASCII_8BIT,
                    message_class=msg_cls,
                    confidence=self._conf_enum(conf_s),
                    confidence_score=conf_s,
                    raw_bits=stream.n_bits,
                    raw_bytes=stream.n_bits // 8,
                    plaintext=txt[:500],
                    language_score=lang_s,
                    entropy_bits=h,
                    notes=f"Extracted via {method_label} | printability={print_score:.2f}",
                ))

            # ── 2b. Caesar cipher ──────────────────────────────────────────
            if len(txt) >= 20:
                caesar = self.cipher.break_caesar(txt)
                if caesar.confidence > 0.4:
                    lang_c = self.ling.score_english(caesar.plaintext)
                    results.append(DecodedMessage(
                        encoding_type=EncodingType.CAESAR,
                        message_class=self.ling.classify_message(caesar.plaintext),
                        confidence=self._conf_enum(caesar.confidence),
                        confidence_score=caesar.confidence,
                        raw_bits=stream.n_bits,
                        raw_bytes=len(txt),
                        plaintext=caesar.plaintext[:500],
                        language_score=lang_c,
                        entropy_bits=self._text_entropy(caesar.plaintext),
                        cipher_key=f"ROT{caesar.key}",
                        key_length=1,
                        notes=f"Caesar shift={caesar.key} chi2={caesar.chi2_score:.3f}",
                    ))

            # ── 2c. Vigenère ───────────────────────────────────────────────
            if len(txt) >= 40:
                vig = self.cipher.break_vigenere(txt, max_key_len=10)
                if vig.confidence > 0.35:
                    lang_v = self.ling.score_english(vig.plaintext)
                    results.append(DecodedMessage(
                        encoding_type=EncodingType.VIGENERE,
                        message_class=self.ling.classify_message(vig.plaintext),
                        confidence=self._conf_enum(vig.confidence),
                        confidence_score=vig.confidence,
                        raw_bits=stream.n_bits,
                        raw_bytes=len(txt),
                        plaintext=vig.plaintext[:500],
                        language_score=lang_v,
                        entropy_bits=self._text_entropy(vig.plaintext),
                        cipher_key=vig.key,
                        key_length=vig.key_length,
                        notes=f"Vigenère key={vig.key!r} IC={vig.ic_score:.4f}",
                    ))

            # ── 2d. XOR cipher ─────────────────────────────────────────────
            xor = self.cipher.break_xor(stream.byte_array, max_key_len=16)
            if xor.confidence > 0.55:
                lang_x = self.ling.score_english(xor.plaintext)
                results.append(DecodedMessage(
                    encoding_type=EncodingType.XOR_CIPHER,
                    message_class=self.ling.classify_message(xor.plaintext),
                    confidence=self._conf_enum(xor.confidence),
                    confidence_score=xor.confidence,
                    raw_bits=stream.n_bits,
                    raw_bytes=stream.n_bits // 8,
                    plaintext=xor.plaintext[:500],
                    language_score=lang_x,
                    entropy_bits=self._text_entropy(xor.plaintext),
                    cipher_key=xor.key,
                    key_length=xor.key_length,
                    notes=f"XOR key=0x{xor.key} len={xor.key_length}",
                ))

            # ── 2e. Base encoding ──────────────────────────────────────────
            strings = self.ascii_frame.extract_strings(stream, min_length=8)
            for s in strings[:5]:
                base_candidates = self.base_enc.detect_and_decode(s)
                for enc_name, decoded_txt, bc_score in base_candidates[:2]:
                    if bc_score > 0.6:
                        lang_b = self.ling.score_english(decoded_txt)
                        enc_type = (EncodingType.BASE64 if "64" in enc_name else
                                    EncodingType.BASE32 if "32" in enc_name else
                                    EncodingType.ASCII_8BIT)
                        results.append(DecodedMessage(
                            encoding_type=enc_type,
                            message_class=self.ling.classify_message(decoded_txt),
                            confidence=self._conf_enum(bc_score),
                            confidence_score=bc_score,
                            raw_bits=len(s) * 8,
                            raw_bytes=len(s),
                            plaintext=decoded_txt[:500],
                            language_score=lang_b,
                            entropy_bits=self._text_entropy(decoded_txt),
                            notes=f"Detected {enc_name} in extracted string",
                        ))

            # ── 2f. RLE ────────────────────────────────────────────────────
            is_rle, rle_conf = self.rle.is_likely_rle(stream.byte_array)
            if is_rle and rle_conf > 0.3:
                decoded_rle = self.rle.decode_rle(stream.byte_array)
                if decoded_rle:
                    try:
                        rle_txt = decoded_rle.decode("latin-1")
                    except Exception:
                        rle_txt = decoded_rle.hex()
                    results.append(DecodedMessage(
                        encoding_type=EncodingType.RUN_LENGTH,
                        message_class=self.ling.classify_message(rle_txt),
                        confidence=self._conf_enum(rle_conf),
                        confidence_score=rle_conf,
                        raw_bits=stream.n_bits,
                        raw_bytes=len(decoded_rle),
                        plaintext=rle_txt[:500],
                        entropy_bits=self._text_entropy(rle_txt),
                        notes=f"RLE decoded: {len(stream.byte_array)}→{len(decoded_rle)} bytes",
                    ))

        # ── 3. Morse code ──────────────────────────────────────────────────
        is_morse, morse_conf = self.morse.detect_morse(signal, self.sr)
        if is_morse:
            morse_text, morse_str, decode_conf = self.morse.decode(signal, self.sr)
            if morse_text:
                lang_m = self.ling.score_english(morse_text)
                combined = float((morse_conf + decode_conf) / 2)
                results.append(DecodedMessage(
                    encoding_type=EncodingType.MORSE,
                    message_class=self.ling.classify_message(morse_text),
                    confidence=self._conf_enum(combined),
                    confidence_score=combined,
                    raw_bits=len(morse_str),
                    raw_bytes=len(morse_text),
                    plaintext=morse_text,
                    language_score=lang_m,
                    entropy_bits=self._text_entropy(morse_text),
                    notes=f"Morse: {morse_str[:100]}",
                ))

        # ── 4. Prime / ARIRAL sequence ─────────────────────────────────────
        prime_seq, prime_conf = self.prime_dec.extract_prime_sequence(signal, self.sr)
        if prime_seq and prime_conf > 0.15:
            symbols, phrase, phrase_conf = self.prime_dec.decode_sequence(prime_seq)
            combined_pc = float((prime_conf + phrase_conf) / 2)
            plain_ariral = " | ".join(symbols[:20])
            results.append(DecodedMessage(
                encoding_type=EncodingType.PRIME_SEQUENCE,
                message_class=MessageClass.ARIRAL_LANGUAGE if phrase_conf > 0.3 else MessageClass.MATHEMATICAL,
                confidence=self._conf_enum(combined_pc),
                confidence_score=combined_pc,
                raw_bits=len(prime_seq),
                raw_bytes=len(prime_seq),
                plaintext=plain_ariral,
                ariral_symbols=symbols,
                ariral_phrase=phrase,
                entropy_bits=self._text_entropy(plain_ariral),
                notes=f"Primes: {prime_seq[:10]} | Phrase conf={phrase_conf:.2f}",
            ))

        # ── 5. Fibonacci word ──────────────────────────────────────────────
        fib_text, fib_conf = self.prime_dec.fibonacci_word_decode(stream_ask)
        if fib_conf > 0.3 and fib_text:
            results.append(DecodedMessage(
                encoding_type=EncodingType.FIBONACCI_WORD,
                message_class=MessageClass.ARIRAL_LANGUAGE if fib_conf > 0.5 else MessageClass.MATHEMATICAL,
                confidence=self._conf_enum(fib_conf),
                confidence_score=fib_conf,
                raw_bits=len(fib_text),
                raw_bytes=len(fib_text) // 8,
                plaintext=fib_text[:200],
                entropy_bits=self._text_entropy(fib_text),
                notes=f"Fibonacci word density={np.mean(stream_ask.bits):.3f} (expected φ⁻¹=0.618)",
            ))

        # ── 6. Spectral steganography ──────────────────────────────────────
        if waterfall is not None:
            stego_stream = self.extractor.extract_spectral_lsb(waterfall)
            if stego_stream.n_bits > 16:
                stego_txt, stego_score = self.ascii_frame.decode(stego_stream)
                if stego_score > 0.45:
                    lang_sg = self.ling.score_english(stego_txt)
                    results.append(DecodedMessage(
                        encoding_type=EncodingType.STEGANOGRAPHIC,
                        message_class=self.ling.classify_message(stego_txt),
                        confidence=self._conf_enum(stego_score),
                        confidence_score=stego_score,
                        raw_bits=stego_stream.n_bits,
                        raw_bytes=stego_stream.n_bits // 8,
                        plaintext=stego_txt[:300],
                        language_score=lang_sg,
                        entropy_bits=self._text_entropy(stego_txt),
                        notes="LSB extracted from spectrogram power matrix",
                    ))

        # ── HMAC tag all results ───────────────────────────────────────────
        for msg in results:
            payload = f"{msg.uid}{msg.encoding_type.value}{msg.plaintext[:50]}"
            msg.hmac_tag = hmac.new(SECRET_KEY, payload.encode(),
                                     hashlib.sha256).hexdigest()[:16].upper()
            msg.checksum_valid = True

        # Sort by confidence descending
        results.sort(key=lambda m: -m.confidence_score)
        return results

    @staticmethod
    def _conf_enum(score: float) -> DecodingConfidence:
        if score > 0.90: return DecodingConfidence.DEFINITIVE
        if score > 0.70: return DecodingConfidence.HIGH
        if score > 0.50: return DecodingConfidence.MODERATE
        if score > 0.20: return DecodingConfidence.SPECULATIVE
        return DecodingConfidence.FAILED

    @staticmethod
    def _text_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        probs  = np.array([v / len(text) for v in counts.values()])
        return float(-np.sum(probs * np.log2(probs + 1e-30)))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ─────────────────────────────────────────────────────────────────────────────

class CryptoVisualizer:
    PAL = {
        "bg":   "#030d07", "fg":   "#00ff88", "dim":  "#004422",
        "acc":  "#00ffcc", "warn": "#ff8800", "dng":  "#ff2222",
        "grid": "#001a0e", "axis": "#335544", "gold": "#ffcc00",
        "purp": "#8844ff",
    }

    def _style(self, ax, title=""):
        ax.set_facecolor(self.PAL["bg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(self.PAL["dim"])
        ax.tick_params(colors=self.PAL["axis"], labelsize=6)
        ax.grid(True, color=self.PAL["grid"], alpha=0.4, ls=":")
        if title:
            ax.set_title(title, color=self.PAL["fg"], fontsize=7.5,
                         loc="left", pad=3, fontfamily="monospace")

    def plot_frequency_analysis(self, text: str,
                                 figsize=(10, 3.5)) -> plt.Figure:
        analyser = CipherAnalyser()
        prof = analyser.frequency_profile(text)
        if not prof.symbol_counts:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO TEXT DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=11)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, (ax_obs, ax_ref) = plt.subplots(1, 2, figsize=figsize,
                                              facecolor=self.PAL["bg"])
        letters   = list(string.ascii_uppercase)
        obs_freq  = np.array([prof.symbol_counts.get(l, 0) / max(prof.total_symbols, 1)
                               for l in letters])
        eng_freq  = np.array([ENGLISH_FREQ.get(l, 0.001) for l in letters])

        ax_obs.bar(letters, obs_freq * 100, color=self.PAL["fg"], alpha=0.82, width=0.85)
        self._style(ax_obs, f"OBSERVED FREQUENCY (IC={prof.ic:.4f})")
        ax_obs.set_xlabel("Letter", fontsize=6, color=self.PAL["fg"])
        ax_obs.set_ylabel("Frequency %", fontsize=6, color=self.PAL["fg"])
        ax_obs.tick_params(axis="x", labelsize=5)

        ax_ref.bar(letters, eng_freq * 100, color=self.PAL["acc"], alpha=0.75, width=0.85)
        self._style(ax_ref, f"ENGLISH REFERENCE (χ²={prof.chi2_vs_english:.3f})")
        ax_ref.set_xlabel("Letter", fontsize=6, color=self.PAL["fg"])
        ax_ref.tick_params(axis="x", labelsize=5)

        plt.tight_layout(pad=0.4)
        return fig

    def plot_entropy_timeline(self, data: bytes,
                               figsize=(10, 2.8)) -> plt.Figure:
        ent_dec = EntropyDecomposer()
        timeline = ent_dec.entropy_timeline(data, window=64)
        if len(timeline) < 2:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "INSUFFICIENT DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=10)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        x = np.arange(len(timeline)) * 32      # byte offset (stride = 32)
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        ax.plot(x, timeline, color=self.PAL["fg"], lw=0.9, alpha=0.9)
        ax.fill_between(x, timeline, 0, alpha=0.12, color=self.PAL["fg"])
        ax.axhline(7.5, color=self.PAL["dng"],  lw=0.7, ls="--", alpha=0.7,
                    label="Encrypted (>7.5)")
        ax.axhline(6.0, color=self.PAL["warn"], lw=0.7, ls="--", alpha=0.7,
                    label="Compressed (>6.0)")
        ax.axhline(3.0, color=self.PAL["acc"],  lw=0.7, ls=":", alpha=0.5,
                    label="Structured (<3.0)")
        ax.set_ylim(0, 8.5)
        self._style(ax, "ENTROPY TIMELINE (byte offset)")
        ax.set_xlabel("Byte Offset", fontsize=6, color=self.PAL["fg"])
        ax.set_ylabel("H (bits/byte)", fontsize=6, color=self.PAL["fg"])
        ax.legend(fontsize=5.5, facecolor=self.PAL["bg"],
                  edgecolor=self.PAL["dim"], labelcolor=self.PAL["fg"])
        plt.tight_layout(pad=0.4)
        return fig

    def plot_bit_stream(self, stream: BinaryStream,
                         max_bits: int = 512,
                         figsize=(10, 1.8)) -> plt.Figure:
        bits = stream.bits[:max_bits]
        if len(bits) == 0:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        x = np.arange(len(bits))
        ax.step(x, bits, color=self.PAL["fg"], lw=0.7, alpha=0.9, where="post")
        ax.fill_between(x, bits, 0, alpha=0.2, color=self.PAL["fg"], step="post")
        ax.set_ylim(-0.2, 1.3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=6, color=self.PAL["axis"])
        self._style(ax, f"BIT STREAM — {stream.method} | {stream.n_bits} bits | "
                         f"rate={stream.bit_rate_bps:.0f} bps | "
                         f"SNR={stream.extraction_snr:.1f} | "
                         f"SYNC={'✓' if stream.sync_word_found else '—'}")
        ax.set_xlabel("Bit Index", fontsize=6, color=self.PAL["fg"])
        plt.tight_layout(pad=0.3)
        return fig

    def plot_ariral_sequence(self, primes: List[int], symbols: List[str],
                              phrase: str, figsize=(10, 3.0)) -> plt.Figure:
        if not primes:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO PRIME SEQUENCE DETECTED",
                    transform=ax.transAxes, ha="center",
                    color=self.PAL["dim"], fontsize=10)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, (ax_seq, ax_info) = plt.subplots(1, 2, figsize=figsize,
                                               facecolor=self.PAL["bg"],
                                               gridspec_kw={"width_ratios": [2, 1]})
        x = np.arange(len(primes))
        ax_seq.stem(x, primes,
                     linefmt=self.PAL["fg"],
                     markerfmt="o",
                     basefmt=self.PAL["dim"])
        for i, (xi, p) in enumerate(zip(x, primes)):
            sym = ARIRAL_SYMBOL_MAP.get(p, "?")
            ax_seq.text(xi, p + max(primes) * 0.04, sym.split()[0],
                         ha="center", va="bottom",
                         color=self.PAL["gold"], fontsize=7,
                         fontfamily="monospace")
        self._style(ax_seq, "ARIRAL PRIME SEQUENCE")
        ax_seq.set_xlabel("Position", fontsize=6, color=self.PAL["fg"])
        ax_seq.set_ylabel("Prime Value", fontsize=6, color=self.PAL["fg"])

        ax_info.axis("off")
        ax_info.set_facecolor(self.PAL["bg"])
        info_lines = [
            ("SEQUENCE", " ".join(str(p) for p in primes[:12])),
            ("SYMBOLS",  " ".join(s.split()[0] for s in symbols[:12])),
            ("PHRASE",   phrase[:50] if phrase else "UNMATCHED"),
            ("N PRIMES", str(len(primes))),
            ("MAX",      str(max(primes))),
        ]
        for i, (label, val) in enumerate(info_lines):
            col = self.PAL["gold"] if label == "PHRASE" else self.PAL["fg"]
            ax_info.text(0.02, 1.0 - i * 0.18, f"{label:<9} {val}",
                          transform=ax_info.transAxes,
                          color=col, fontsize=6.5, fontfamily="monospace",
                          va="top")
        self._style(ax_info, "ARIRAL DECODE")
        plt.tight_layout(pad=0.4)
        return fig

    def plot_candidate_ranking(self, messages: List[DecodedMessage],
                                figsize=(9, 4.0)) -> plt.Figure:
        if not messages:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO DECODE CANDIDATES", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=11)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        n = min(len(messages), 8)
        msgs = messages[:n]
        labels = [f"{m.encoding_type.value[:12]}\n({m.confidence.value[:4]})"
                  for m in msgs]
        scores = [m.confidence_score for m in msgs]
        lang_s = [m.language_score for m in msgs]
        cols   = [self.PAL["dng"]  if s > 0.85 else
                   self.PAL["warn"] if s > 0.65 else
                   self.PAL["fg"]   if s > 0.45 else
                   self.PAL["dim"]
                   for s in scores]

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
        x = np.arange(n)
        bars = ax.bar(x, scores, color=cols, alpha=0.82, width=0.55, label="Conf")
        ax.plot(x, lang_s, color=self.PAL["acc"], lw=1.0, marker="s",
                 ms=4, ls="--", label="Language Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, color=self.PAL["axis"])
        ax.set_ylim(0, 1.1)
        ax.axhline(0.7, color=self.PAL["warn"], lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0.9, color=self.PAL["dng"],  lw=0.7, ls=":", alpha=0.5)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     s + 0.02, f"{s:.2f}",
                     ha="center", va="bottom", fontsize=5.5,
                     color=self.PAL["acc"], fontfamily="monospace")
        self._style(ax, "DECODE CANDIDATE RANKING — CONFIDENCE vs LANGUAGE SCORE")
        ax.set_ylabel("Score (0–1)", fontsize=6, color=self.PAL["fg"])
        ax.legend(fontsize=6, facecolor=self.PAL["bg"],
                  edgecolor=self.PAL["dim"], labelcolor=self.PAL["fg"])
        plt.tight_layout(pad=0.4)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def crypto_decoder_page():
    from signal_engine import (
        init_session_state, SignalGenerator, SignalClass, _generate_for_class
    )
    init_session_state()

    st.markdown("""
    <style>
    .cry-header {
        font-family:'Courier New',monospace;
        color:#00ff88;
        font-size:0.78rem;
        letter-spacing:0.14em;
        border-bottom:1px solid #00ff4430;
        padding-bottom:0.4rem;
        margin-bottom:1rem;
    }
    .cry-label {
        font-family:'Courier New',monospace;
        color:#88ffcc;
        font-size:0.70rem;
        letter-spacing:0.09em;
        margin-top:0.5rem;
        margin-bottom:0.2rem;
    }
    .ariral-phrase {
        background:#001a0e;
        border:1px solid #ffcc00;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.78rem;
        color:#ffcc00;
        letter-spacing:0.10em;
        margin-bottom:0.5rem;
    }
    .decode-box {
        background:#050f08;
        border:1px solid #00ff4440;
        border-radius:2px;
        padding:0.5rem 0.8rem;
        font-family:'Courier New',monospace;
        font-size:0.67rem;
        color:#88ffcc;
        white-space:pre-wrap;
        overflow-x:auto;
        margin-bottom:0.4rem;
        max-height:160px;
    }
    .hex-dump {
        background:#030d07;
        border:1px solid #004422;
        border-radius:2px;
        padding:0.4rem 0.6rem;
        font-family:'Courier New',monospace;
        font-size:0.60rem;
        color:#444;
        white-space:pre;
        overflow-x:auto;
        max-height:120px;
    }
    .cipher-result {
        background:#0a1f0a;
        border:1px solid #00ff4450;
        border-radius:2px;
        padding:0.4rem 0.7rem;
        font-family:'Courier New',monospace;
        font-size:0.66rem;
        color:#00ffcc;
        margin-bottom:0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="cry-header">[ CRYPTOGRAPHY & MESSAGE DECODING ENGINE — SIGNAL INTELLIGENCE ]</div>',
        unsafe_allow_html=True)

    if "decoder_archive" not in st.session_state:
        st.session_state.decoder_archive = []
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = []

    gen:  SignalGenerator = st.session_state.generator
    viz = CryptoVisualizer()

    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown('<div class="cry-label">— SIGNAL INPUT —</div>', unsafe_allow_html=True)
        sig_type  = st.selectbox("Signal Class",
                                  [c.value for c in SignalClass])
        snr_db    = st.slider("SNR (dB)", -5.0, 40.0, 20.0, 1.0)
        duration  = st.slider("Duration (s)", 0.5, 15.0, 6.0, 0.5)
        dm_val    = st.number_input("DM (pc·cm⁻³)", 0.0, 2000.0, 0.0, step=1.0)

        st.markdown('<div class="cry-label">— EXTRACTION CONFIG —</div>', unsafe_allow_html=True)
        bit_rate_override = st.number_input("Bit Rate Override (bps, 0=auto)",
                                              0, 115200, 0, step=100)
        use_waterfall = st.checkbox("Include spectral steganography", value=True)
        show_hex      = st.checkbox("Show hex dump", value=True)

        st.markdown('<div class="cry-label">— MANUAL CIPHER TOOL —</div>', unsafe_allow_html=True)
        raw_text = st.text_area("Paste ciphertext for analysis", height=80,
                                  placeholder="Enter any text for cipher analysis…")
        cipher_method = st.selectbox("Force cipher method",
                                      ["AUTO", "CAESAR", "VIGENERE", "XOR", "BASE64", "FREQUENCY"])

        st.markdown('<div class="cry-label">— ARIRAL QUERY —</div>', unsafe_allow_html=True)
        prime_input = st.text_input("Prime sequence (space-separated)", "3 5 41 67")

    with col_main:
        tabs = st.tabs(["[ DECODE PIPELINE ]",
                        "[ CIPHER ANALYSIS ]",
                        "[ ARIRAL DECODER ]",
                        "[ ENTROPY ]",
                        "[ ARCHIVE ]"])

        # ── Tab 1: Full decode pipeline ────────────────────────────────────
        with tabs[0]:
            if st.button("▶ DECODE SIGNAL", use_container_width=True):
                sig_class = SignalClass(sig_type)
                with st.spinner("[ EXTRACTING BITS … RUNNING CIPHER SUITE … ]"):
                    t, signal = _generate_for_class(gen, sig_class, duration, snr_db, dm_val)
                    decoder = MasterDecoder(sample_rate=2e6)

                    # Bit rate
                    if bit_rate_override > 0:
                        bps = float(bit_rate_override)
                    else:
                        bps = decoder.extractor.detect_bit_rate(signal)

                    # Waterfall for stego
                    waterfall = None
                    if use_waterfall:
                        from spectral_analyzer import SpectralAnalyzer
                        sa = SpectralAnalyzer(sample_rate=2e6)
                        ds = sa.build_dynamic_spectrum(signal, n_fft=256, hop=64)
                        waterfall = ds.power_db_matrix

                    messages = decoder.decode_signal(signal, waterfall)
                    st.session_state.current_messages = messages
                    st.session_state.decoder_archive.extend(messages)

                    # Primary stream for display
                    stream = decoder.extractor.extract_ask(signal, bps)

                # Bit stream plot
                fig_bits = viz.plot_bit_stream(stream)
                st.pyplot(fig_bits, use_container_width=True)
                plt.close(fig_bits)

                st.markdown(
                    f'<div class="cry-label">BIT RATE: {bps:.0f} bps | '
                    f'BITS: {stream.n_bits} | '
                    f'BYTES: {stream.n_bits//8} | '
                    f'SYNC: {"✓" if stream.sync_word_found else "—"} | '
                    f'SNR: {stream.extraction_snr:.2f}</div>',
                    unsafe_allow_html=True)

                # Hex dump
                if show_hex and stream.byte_array:
                    st.markdown('<div class="cry-label">— HEX DUMP (first 256 bytes) —</div>',
                                unsafe_allow_html=True)
                    hex_str = BinaryStream(stream.bits, bps, 0, "").hex_dump()[:2000]
                    st.markdown(f'<div class="hex-dump">{hex_str}</div>',
                                unsafe_allow_html=True)

                # Candidate ranking
                if messages:
                    fig_rank = viz.plot_candidate_ranking(messages)
                    st.pyplot(fig_rank, use_container_width=True)
                    plt.close(fig_rank)

                    # Top result
                    top = messages[0]
                    st.markdown('<div class="cry-label">— BEST DECODE RESULT —</div>',
                                unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Encoding",   top.encoding_type.value.replace("_", " "))
                    m2.metric("Confidence", f"{top.confidence_score*100:.0f}%")
                    m3.metric("Class",      top.message_class.value.replace("_", " "))
                    m4.metric("Entropy",    f"{top.entropy_bits:.3f} bits")

                    st.markdown(
                        f'<div class="decode-box">{top.plaintext[:400] if top.plaintext else "(empty)"}</div>',
                        unsafe_allow_html=True)

                    if top.ariral_phrase:
                        st.markdown(
                            f'<div class="ariral-phrase">◈ ARIRAL: {top.ariral_phrase}</div>',
                            unsafe_allow_html=True)

                    if top.cipher_key:
                        st.markdown(
                            f'<div class="cipher-result">KEY: {top.cipher_key} | '
                            f'KEY LEN: {top.key_length} | '
                            f'HMAC: {top.hmac_tag}</div>',
                            unsafe_allow_html=True)

                    # All candidates table
                    with st.expander("[ ALL DECODE CANDIDATES ]"):
                        rows_df = pd.DataFrame([m.to_row() for m in messages])
                        st.dataframe(rows_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown('<div class="cry-label">NO DECODABLE CONTENT FOUND.</div>',
                                unsafe_allow_html=True)

        # ── Tab 2: Manual cipher analysis ──────────────────────────────────
        with tabs[1]:
            text_to_analyse = raw_text.strip() if raw_text.strip() else \
                "KHOOR ZRUOG"  # default: ROT3 "HELLO WORLD"
            analyser = CipherAnalyser()

            if st.button("▶ ANALYSE CIPHERTEXT", use_container_width=True):
                with st.spinner("[ RUNNING CRYPTANALYSIS … ]"):
                    # Frequency plot
                    fig_freq = viz.plot_frequency_analysis(text_to_analyse)
                    st.pyplot(fig_freq, use_container_width=True)
                    plt.close(fig_freq)

                    prof = analyser.frequency_profile(text_to_analyse)
                    fp1, fp2, fp3, fp4 = st.columns(4)
                    fp1.metric("IC",      f"{prof.ic:.4f}")
                    fp2.metric("Entropy", f"{prof.entropy:.3f} bits")
                    fp3.metric("χ²",      f"{prof.chi2_vs_english:.4f}")
                    fp4.metric("Natural", "YES" if prof.is_likely_natural else "NO")

                    if cipher_method in ("AUTO", "CAESAR"):
                        caesar = analyser.break_caesar(text_to_analyse)
                        st.markdown(f'<div class="cipher-result">'
                                     f'CAESAR ROT{caesar.key}: conf={caesar.confidence:.2f} | '
                                     f'IC={caesar.ic_score:.4f}<br>{caesar.plaintext[:200]}'
                                     f'</div>', unsafe_allow_html=True)

                    if cipher_method in ("AUTO", "VIGENERE") and len(text_to_analyse) >= 20:
                        key_lens = analyser.friedman_key_length(text_to_analyse)
                        st.markdown('<div class="cry-label">— FRIEDMAN KEY LENGTH CANDIDATES —</div>',
                                     unsafe_allow_html=True)
                        kl_df = pd.DataFrame(key_lens[:5], columns=["Key Length", "Average IC"])
                        kl_df["IC Δ from English"] = (kl_df["Average IC"] - 0.0667).abs().round(4)
                        st.dataframe(kl_df, use_container_width=True, hide_index=True)

                        if len(text_to_analyse) >= 40:
                            vig = analyser.break_vigenere(text_to_analyse, max_key_len=12)
                            st.markdown(
                                f'<div class="cipher-result">'
                                f'VIGENÈRE key={vig.key!r} len={vig.key_length} '
                                f'conf={vig.confidence:.2f} IC={vig.ic_score:.4f}<br>'
                                f'{vig.plaintext[:200]}'
                                f'</div>', unsafe_allow_html=True)

                    if cipher_method in ("AUTO", "BASE64", "FREQUENCY"):
                        base_det = BaseEncodingDetector()
                        candidates = base_det.detect_and_decode(text_to_analyse)
                        if candidates:
                            st.markdown('<div class="cry-label">— BASE ENCODING DETECTED —</div>',
                                         unsafe_allow_html=True)
                            for enc, decoded, score in candidates[:3]:
                                st.markdown(
                                    f'<div class="cipher-result">'
                                    f'{enc}: printability={score:.2f}<br>{decoded[:150]}'
                                    f'</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="cry-label">NO BASE ENCODING DETECTED.</div>',
                                         unsafe_allow_html=True)

        # ── Tab 3: ARIRAL decoder ───────────────────────────────────────────
        with tabs[2]:
            if st.button("▶ DECODE ARIRAL SEQUENCE", use_container_width=True):
                try:
                    input_primes = [int(p.strip()) for p in prime_input.split()
                                    if p.strip().isdigit()]
                except ValueError:
                    input_primes = []

                if not input_primes:
                    input_primes = [3, 5, 41, 67]

                psd = PrimeSequenceDecoder()
                symbols, phrase, phrase_conf = psd.decode_sequence(input_primes)

                fig_ar = viz.plot_ariral_sequence(input_primes, symbols, phrase)
                st.pyplot(fig_ar, use_container_width=True)
                plt.close(fig_ar)

                if phrase:
                    st.markdown(
                        f'<div class="ariral-phrase">◈ TRANSLATION: {phrase}<br>'
                        f'   CONFIDENCE: {phrase_conf*100:.0f}%</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="cry-label">NO KNOWN ARIRAL PHRASE MATCHED. SEQUENCE RECORDED.</div>',
                        unsafe_allow_html=True)

                # Symbol table
                symbol_rows = []
                for p, sym in zip(input_primes, symbols):
                    symbol_rows.append({
                        "Prime":  p,
                        "Symbol": sym,
                        "Nearest Known": min(ARIRAL_PRIMES, key=lambda x: abs(x - p)),
                    })
                st.dataframe(pd.DataFrame(symbol_rows),
                             use_container_width=True, hide_index=True)

                # Known phrases reference
                with st.expander("[ KNOWN ARIRAL PHRASE CATALOG ]"):
                    phrase_rows = [{"Primes": str(pp), "Translation": txt}
                                   for pp, txt in KNOWN_ARIRAL_PHRASES]
                    st.dataframe(pd.DataFrame(phrase_rows),
                                 use_container_width=True, hide_index=True)

                # Full ARIRAL symbol table
                with st.expander("[ ARIRAL SYMBOL MAP ]"):
                    sym_rows = [{"Prime": p, "Symbol": s}
                                for p, s in ARIRAL_SYMBOL_MAP.items()]
                    st.dataframe(pd.DataFrame(sym_rows),
                                 use_container_width=True, hide_index=True)

        # ── Tab 4: Entropy analysis ─────────────────────────────────────────
        with tabs[3]:
            if st.button("▶ ENTROPY ANALYSIS", use_container_width=True):
                sig_class = SignalClass(sig_type)
                with st.spinner("[ COMPUTING ENTROPY … ]"):
                    _, signal = _generate_for_class(gen, sig_class, duration, snr_db, dm_val)
                    decoder2 = MasterDecoder(sample_rate=2e6)
                    stream2  = decoder2.extractor.extract_ask(signal, 1200.0)
                    data_bytes = stream2.byte_array

                if data_bytes:
                    ent_dec = EntropyDecomposer()
                    ent_info = ent_dec.compute(data_bytes)

                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Byte Entropy",   f"{ent_info['byte_entropy']:.4f} bits")
                    e2.metric("Bit Entropy",    f"{ent_info['bit_entropy']:.4f} bits")
                    e3.metric("Unique Bytes",   ent_info['n_unique_bytes'])
                    e4.metric("Classification", ent_info['classification'][:14])

                    ec1, ec2 = st.columns(2)
                    ec1.metric("Byte Correlation", f"{ent_info['byte_correlation']:.4f}")
                    ec2.metric("Likely Cipher", "YES" if ent_info['likely_cipher'] else "NO")

                    fig_ent = viz.plot_entropy_timeline(data_bytes)
                    st.pyplot(fig_ent, use_container_width=True)
                    plt.close(fig_ent)

                    ent_df = pd.DataFrame([{"Metric": k, "Value": str(v)}
                                            for k, v in ent_info.items()])
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown('<div class="cry-label">INSUFFICIENT DATA.</div>',
                                unsafe_allow_html=True)

        # ── Tab 5: Archive ──────────────────────────────────────────────────
        with tabs[4]:
            archive = st.session_state.decoder_archive
            if archive:
                arch_df = pd.DataFrame([m.to_row() for m in archive[-50:]])
                st.dataframe(arch_df, use_container_width=True, hide_index=True)

                # Summary stats
                classes = Counter(m.message_class.value for m in archive)
                encodings = Counter(m.encoding_type.value for m in archive)
                ac1, ac2 = st.columns(2)
                ac1.markdown('<div class="cry-label">— MESSAGE CLASSES —</div>', unsafe_allow_html=True)
                ac1.dataframe(
                    pd.DataFrame(classes.most_common(), columns=["Class", "Count"]),
                    use_container_width=True, hide_index=True)
                ac2.markdown('<div class="cry-label">— ENCODING TYPES —</div>', unsafe_allow_html=True)
                ac2.dataframe(
                    pd.DataFrame(encodings.most_common(), columns=["Encoding", "Count"]),
                    use_container_width=True, hide_index=True)

                ariral_msgs = [m for m in archive if m.message_class == MessageClass.ARIRAL_LANGUAGE]
                if ariral_msgs:
                    st.markdown('<div class="cry-label">— ARIRAL MESSAGES —</div>',
                                unsafe_allow_html=True)
                    for msg in ariral_msgs[-5:]:
                        st.markdown(
                            f'<div class="ariral-phrase">'
                            f'[{msg.uid}] {msg.ariral_phrase or msg.plaintext[:80]}'
                            f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="cry-label">ARCHIVE EMPTY. RUN DECODE PIPELINE TO POPULATE.</div>',
                    unsafe_allow_html=True)


if __name__ == "__main__":
    crypto_decoder_page()
