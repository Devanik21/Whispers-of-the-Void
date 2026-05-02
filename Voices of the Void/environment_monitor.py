"""
environment_monitor.py — Observatory Environment & Systems Monitor
Voices of the Void | Alpen Signal Observatorium — Dunkeltaler Forest
Station Telemetry, Life Support & Event Management System v2.9.3

Handles: Day/night cycle with astronomical twilight, atmospheric noise modelling
         (ionospheric, tropospheric, galactic), satellite dish health (motor,
         surface, cable, feed horn), power system (generator, solar, battery),
         sleep schedule with fatigue/hallucination mechanics, full inventory
         management (food, drives, repair kits, medicine), random event engine
         (power outages, equipment failure, animal encounters, ARIRAL packages),
         weather simulation (5 states), RFI environment from nearby Schwarztal
         village, maintenance scheduling, station log/journal, environmental
         impact on signal quality, badge/achievement system.

This is your life. These are the systems keeping you alive.
Do not neglect them.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union
)

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL & STATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

STATION_LATITUDE_DEG   =  46.8        # Swiss Alps, Dunkeltaler valley
STATION_LONGITUDE_DEG  =   8.1
STATION_ALTITUDE_M     = 1840.0       # metres above sea level
SOLAR_CONSTANT_WM2     = 1361.0       # solar irradiance at top of atmosphere
DISH_DIAMETER_M        =    8.5       # parabolic dish diameter
DISH_AREA_M2           = math.pi * (DISH_DIAMETER_M / 2) ** 2
SYSTEM_TSYS_K          =   25.0       # system temperature (K)
BOLTZMANN_K            = 1.380649e-23
GENERATOR_CAPACITY_L   =   50.0       # fuel tank (litres)
GENERATOR_BURN_RATE    =    1.2       # litres/hour at full load
SOLAR_PANEL_KW         =    2.4       # installed solar capacity
BATTERY_CAPACITY_KWH   =   20.0      # battery bank
BASE_LOAD_KW           =    1.8       # observatory base power draw

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class WeatherState(Enum):
    CLEAR        = "CLEAR"
    PARTLY_CLOUDY = "PARTLY_CLOUDY"
    OVERCAST     = "OVERCAST"
    FOG          = "FOG"
    STORM        = "STORM"

class DayPhase(Enum):
    NIGHT          = "NIGHT"          # Sun < -18° (astronomical night)
    ASTRO_TWILIGHT = "ASTRO_TWILIGHT" # -18° to -12°
    NAUTICAL_TWIL  = "NAUTICAL_TWIL"  # -12° to -6°
    CIVIL_TWILIGHT = "CIVIL_TWILIGHT" # -6° to 0°
    SUNRISE_SUNSET = "SUNRISE_SUNSET" # 0° ± 0.5°
    DAY            = "DAY"            # Sun > 0°

class EquipmentStatus(Enum):
    NOMINAL    = "NOMINAL"
    DEGRADED   = "DEGRADED"
    CRITICAL   = "CRITICAL"
    OFFLINE    = "OFFLINE"

class EventSeverity(Enum):
    INFO     = 0
    WARNING  = 1
    CRITICAL = 2
    HORROR   = 3

class FatigueLevel(Enum):
    RESTED       = "RESTED"       # 0–20h awake
    TIRED        = "TIRED"        # 20–28h awake
    EXHAUSTED    = "EXHAUSTED"    # 28–36h awake
    DELIRIOUS    = "DELIRIOUS"    # > 36h awake (hallucinations begin)

class PowerSource(Enum):
    SOLAR     = "SOLAR"
    GENERATOR = "GENERATOR"
    BATTERY   = "BATTERY"
    OFFLINE   = "OFFLINE"

class MaintenanceType(Enum):
    DISH_CALIBRATION    = "DISH_CALIBRATION"
    DISH_CLEANING       = "DISH_CLEANING"
    MOTOR_LUBRICATION   = "MOTOR_LUBRICATION"
    CABLE_INSPECTION    = "CABLE_INSPECTION"
    RECEIVER_TUNING     = "RECEIVER_TUNING"
    GENERATOR_SERVICE   = "GENERATOR_SERVICE"
    FILTER_REPLACEMENT  = "FILTER_REPLACEMENT"
    FEED_HORN_ALIGNMENT = "FEED_HORN_ALIGNMENT"

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeatherRecord:
    timestamp:          float
    state:              WeatherState
    temperature_c:      float
    humidity_pct:       float
    wind_speed_ms:      float
    wind_direction_deg: float
    precipitation_mm_h: float
    cloud_cover_pct:    float
    pressure_hpa:       float
    visibility_km:      float
    lightning_active:   bool = False

    @property
    def radio_seeing_factor(self) -> float:
        """
        Signal quality multiplier [0,1] due to weather.
        Rain attenuates high-frequency signals; ionospheric scintillation
        increases at dawn/dusk; fog causes multipath.
        """
        base = 1.0
        if self.state == WeatherState.STORM:
            base = 0.25
        elif self.state == WeatherState.FOG:
            base = 0.55
        elif self.state == WeatherState.OVERCAST:
            base = 0.75
        elif self.state == WeatherState.PARTLY_CLOUDY:
            base = 0.90
        # Wind-induced dish vibration
        wind_deg = max(0.0, 1.0 - self.wind_speed_ms / 25.0)
        # Precipitation: rain scatter (L-band ~0.01 dB/km at 10 mm/h)
        rain_att = math.exp(-0.001 * self.precipitation_mm_h)
        return float(base * wind_deg * rain_att)

    @property
    def atmospheric_noise_k(self) -> float:
        """Atmospheric contribution to system temperature (K)."""
        tau = 0.01 + 0.002 * self.humidity_pct / 100
        if self.state == WeatherState.STORM:
            tau += 0.05
        t_atm = self.temperature_c + 273.15
        return float(t_atm * (1 - math.exp(-tau)))


@dataclass
class DishHealth:
    # Surface: reflectivity degradation from contamination/damage
    surface_reflectivity:   float = 1.0       # 0–1
    surface_contamination:  float = 0.0       # 0–1 (snow, ice, leaves)
    surface_damage:         float = 0.0       # 0–1 (physical)

    # Azimuth motor
    az_motor_health:        float = 1.0       # 0–1
    az_encoder_drift_deg:   float = 0.0       # degrees pointing error
    az_backlash_deg:        float = 0.0       # mechanical backlash

    # Elevation motor
    el_motor_health:        float = 1.0
    el_encoder_drift_deg:   float = 0.0
    el_backlash_deg:        float = 0.0

    # Feed horn
    feed_alignment_deg:     float = 0.0       # degrees off boresight
    feed_impedance_match:   float = 1.0       # 0–1 (1=perfect match)
    polariser_rotation_deg: float = 0.0       # linear pol rotation error

    # Receiver / LNA
    lna_noise_figure_db:    float = 0.5       # dB (baseline ~0.5)
    lna_gain_db:            float = 40.0      # dB
    lna_compression_dbm:    float = -5.0      # 1dB compression point

    # Cable & connector
    cable_loss_db:          float = 0.5       # nominal cable loss
    connector_vswr:         float = 1.05      # voltage standing wave ratio

    @property
    def effective_area_fraction(self) -> float:
        """Aperture efficiency degradation from all mechanical errors."""
        # Surface accuracy: Ruze formula eta_s = exp(-(4π*σ/λ)²)
        sigma_m = self.surface_damage * 0.005  # 5mm RMS at full damage
        lambda_m = 0.21  # 21cm wavelength
        ruze = math.exp(-(4 * math.pi * sigma_m / lambda_m) ** 2)
        # Pointing error (Gaussian beam approx)
        theta_e = math.sqrt(self.az_encoder_drift_deg ** 2 +
                            self.el_encoder_drift_deg ** 2 +
                            self.feed_alignment_deg ** 2)
        theta_3db = 1.22 * (lambda_m / DISH_DIAMETER_M) * 180 / math.pi
        pointing = math.exp(-2.77 * (theta_e / theta_3db) ** 2)
        # Contamination (physical blockage)
        contam = 1.0 - self.surface_contamination * 0.3
        # Feed mismatch loss
        vswr = self.connector_vswr
        mismatch = 1.0 - ((vswr - 1) / (vswr + 1)) ** 2
        return float(ruze * pointing * contam * mismatch *
                     self.surface_reflectivity * self.feed_impedance_match)

    @property
    def system_temp_penalty_k(self) -> float:
        """Additional system temperature from degraded components (K)."""
        # Cable loss adds noise: T_cable = T_phys * (L - 1) / L
        l_lin = 10 ** (self.cable_loss_db / 10)
        t_cable = 290 * (l_lin - 1) / l_lin
        # LNA noise figure
        nf_lin = 10 ** (self.lna_noise_figure_db / 10)
        t_lna = 290 * (nf_lin - 1)
        return float(t_cable + t_lna)

    @property
    def overall_status(self) -> EquipmentStatus:
        eff = self.effective_area_fraction
        if eff > 0.85 and self.lna_noise_figure_db < 1.0:
            return EquipmentStatus.NOMINAL
        elif eff > 0.65:
            return EquipmentStatus.DEGRADED
        elif eff > 0.40:
            return EquipmentStatus.CRITICAL
        return EquipmentStatus.OFFLINE

    def degrade(self, delta_time_h: float, weather: WeatherRecord,
                rng: np.random.Generator) -> None:
        """Apply time-based degradation from weather and normal wear."""
        # Surface contamination from precipitation
        if weather.precipitation_mm_h > 0:
            snow_chance = 1.0 if weather.temperature_c < 0 else 0.0
            ice_chance  = 1.0 if weather.temperature_c < -5 else 0.0
            contam_rate = (0.002 * snow_chance + 0.001 * ice_chance +
                           0.0005 * weather.precipitation_mm_h)
            self.surface_contamination = float(np.clip(
                self.surface_contamination + contam_rate * delta_time_h, 0, 1))

        # Wind damage
        if weather.wind_speed_ms > 15:
            wind_damage = 0.0001 * (weather.wind_speed_ms - 15) ** 2 * delta_time_h
            self.surface_damage = float(np.clip(self.surface_damage + wind_damage, 0, 1))
            # Encoder drift from vibration
            self.az_encoder_drift_deg += float(rng.exponential(0.001 * weather.wind_speed_ms))
            self.el_encoder_drift_deg += float(rng.exponential(0.001 * weather.wind_speed_ms))

        # Thermal cycling — LNA noise figure drift
        temp_stress = abs(weather.temperature_c) / 40.0
        self.lna_noise_figure_db = float(np.clip(
            self.lna_noise_figure_db + rng.normal(0, 0.001 * temp_stress * delta_time_h),
            0.3, 5.0))

        # Motor wear
        self.az_motor_health = float(np.clip(
            self.az_motor_health - rng.exponential(0.0001 * delta_time_h), 0, 1))
        self.el_motor_health = float(np.clip(
            self.el_motor_health - rng.exponential(0.0001 * delta_time_h), 0, 1))

    def repair(self, maintenance_type: MaintenanceType, quality: float = 1.0) -> str:
        """Apply a maintenance action. quality: 0–1 (skill of technician)."""
        restored = []
        if maintenance_type == MaintenanceType.DISH_CLEANING:
            self.surface_contamination = float(np.clip(
                self.surface_contamination - 0.8 * quality, 0, 1))
            restored.append("surface contamination")
        elif maintenance_type == MaintenanceType.DISH_CALIBRATION:
            self.az_encoder_drift_deg *= (1 - 0.9 * quality)
            self.el_encoder_drift_deg *= (1 - 0.9 * quality)
            self.feed_alignment_deg   *= (1 - 0.85 * quality)
            restored.append("encoder drift, feed alignment")
        elif maintenance_type == MaintenanceType.MOTOR_LUBRICATION:
            self.az_motor_health = float(np.clip(self.az_motor_health + 0.3 * quality, 0, 1))
            self.el_motor_health = float(np.clip(self.el_motor_health + 0.3 * quality, 0, 1))
            self.az_backlash_deg *= (1 - 0.5 * quality)
            self.el_backlash_deg *= (1 - 0.5 * quality)
            restored.append("motor health, backlash")
        elif maintenance_type == MaintenanceType.CABLE_INSPECTION:
            self.cable_loss_db    = float(max(0.5, self.cable_loss_db - 0.2 * quality))
            self.connector_vswr   = float(max(1.02, self.connector_vswr - 0.05 * quality))
            restored.append("cable loss, VSWR")
        elif maintenance_type == MaintenanceType.RECEIVER_TUNING:
            self.lna_noise_figure_db = float(max(0.3, self.lna_noise_figure_db - 0.5 * quality))
            self.lna_gain_db         = float(np.clip(self.lna_gain_db + 2 * quality, 35, 45))
            self.feed_impedance_match = float(min(1.0, self.feed_impedance_match + 0.1 * quality))
            restored.append("LNA NF, gain, impedance")
        elif maintenance_type == MaintenanceType.FEED_HORN_ALIGNMENT:
            self.feed_alignment_deg   = float(max(0, self.feed_alignment_deg - 1.5 * quality))
            self.polariser_rotation_deg *= (1 - 0.7 * quality)
            restored.append("feed horn, polariser")
        elif maintenance_type == MaintenanceType.FILTER_REPLACEMENT:
            self.lna_compression_dbm = float(min(-3, self.lna_compression_dbm + 2 * quality))
            restored.append("filter compression")
        elif maintenance_type == MaintenanceType.GENERATOR_SERVICE:
            restored.append("generator (external)")
        return f"Maintenance: {maintenance_type.value} — Restored: {', '.join(restored)}"


@dataclass
class PowerSystem:
    generator_fuel_l:   float = GENERATOR_CAPACITY_L
    battery_charge_kwh: float = BATTERY_CAPACITY_KWH * 0.8
    solar_irradiance:   float = 0.0    # W/m²
    active_source:      PowerSource = PowerSource.GENERATOR
    generator_running:  bool = True
    grid_tied:          bool = False
    load_kw:            float = BASE_LOAD_KW

    @property
    def battery_fraction(self) -> float:
        return float(np.clip(self.battery_charge_kwh / BATTERY_CAPACITY_KWH, 0, 1))

    @property
    def solar_power_kw(self) -> float:
        return float(self.solar_irradiance / 1000 * SOLAR_PANEL_KW * 0.18)  # 18% efficiency

    @property
    def is_powered(self) -> bool:
        return (self.generator_running and self.generator_fuel_l > 0) or \
               self.battery_fraction > 0.05 or self.solar_power_kw > 0.5

    def update(self, delta_h: float, weather: WeatherRecord,
               day_phase: DayPhase) -> Dict[str, Any]:
        """Simulate power system for delta_h hours. Returns events dict."""
        events: Dict[str, Any] = {}

        # Solar generation
        if day_phase == DayPhase.DAY:
            cloud_factor = 1.0 - weather.cloud_cover_pct / 100 * 0.8
            self.solar_irradiance = float(SOLAR_CONSTANT_WM2 *
                                           math.sin(math.pi * 0.5) *  # noon approx
                                           cloud_factor)
        elif day_phase in (DayPhase.CIVIL_TWILIGHT, DayPhase.SUNRISE_SUNSET):
            self.solar_irradiance = float(SOLAR_CONSTANT_WM2 * 0.15)
        else:
            self.solar_irradiance = 0.0

        solar_gen_kwh = self.solar_power_kw * delta_h

        # Generator consumption
        if self.generator_running and self.generator_fuel_l > 0:
            fuel_consumed = GENERATOR_BURN_RATE * delta_h
            self.generator_fuel_l = float(max(0, self.generator_fuel_l - fuel_consumed))
            gen_power_kwh = 3.5 * delta_h   # 3.5 kW generator
            if self.generator_fuel_l == 0:
                self.generator_running = False
                events["generator_stall"] = True
        else:
            gen_power_kwh = 0.0

        # Battery balance
        net_kwh = solar_gen_kwh + gen_power_kwh - self.load_kw * delta_h
        self.battery_charge_kwh = float(
            np.clip(self.battery_charge_kwh + net_kwh, 0, BATTERY_CAPACITY_KWH))

        # Source priority: solar > generator > battery
        if self.solar_power_kw >= self.load_kw:
            self.active_source = PowerSource.SOLAR
        elif self.generator_running:
            self.active_source = PowerSource.GENERATOR
        elif self.battery_fraction > 0.05:
            self.active_source = PowerSource.BATTERY
        else:
            self.active_source = PowerSource.OFFLINE
            events["power_outage"] = True

        if self.generator_fuel_l < 5:
            events["low_fuel"] = True

        return events

    def refuel(self, litres: float = 20.0) -> str:
        added = min(litres, GENERATOR_CAPACITY_L - self.generator_fuel_l)
        self.generator_fuel_l += added
        return f"Added {added:.1f} L → {self.generator_fuel_l:.1f} L total"

    def toggle_generator(self) -> bool:
        if not self.generator_running and self.generator_fuel_l > 0.1:
            self.generator_running = True
        else:
            self.generator_running = False
        return self.generator_running


@dataclass
class Inventory:
    # Food
    food_rations:    int = 14     # days of food
    coffee_packs:    int = 20     # +5h fatigue resistance each
    energy_drinks:   int = 5      # +8h fatigue resistance, +stress
    vitamin_packs:   int = 10     # +2h fatigue resistance, health

    # Maintenance supplies
    lubricant_cans:  int = 3
    cleaning_kits:   int = 4
    cable_spools_m:  int = 50     # metres of cable
    spare_fuses:     int = 10
    filter_modules:  int = 2
    epoxy_packs:     int = 5      # for dish surface repair

    # Computing & storage
    blank_drives:    int = 6      # 512MB each
    usb_adapters:    int = 4
    sd_cards:        int = 8

    # Medical
    painkillers:     int = 20
    bandages:        int = 10
    antibiotics:     int = 4

    # Special items (VotV themed)
    shrimp_packs:    int = 0      # ARIRAL gifts
    void_fragments:  int = 0      # RESTRICTED — destroy immediately
    anomalous_notes: int = 0      # from entity events

    def consume(self, item: str, qty: int = 1) -> bool:
        """Consume qty of item. Returns True if successful."""
        current = getattr(self, item, 0)
        if current >= qty:
            setattr(self, item, current - qty)
            return True
        return False

    def add(self, item: str, qty: int = 1) -> None:
        current = getattr(self, item, 0)
        setattr(self, item, current + qty)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def critical_shortages(self) -> List[str]:
        """Return list of critically low items."""
        shortages = []
        if self.food_rations < 3:     shortages.append("FOOD (<3 days)")
        if self.blank_drives < 2:     shortages.append("DRIVES (<2)")
        if self.lubricant_cans < 1:   shortages.append("LUBRICANT")
        if self.cleaning_kits < 1:    shortages.append("CLEANING")
        return shortages


@dataclass
class SleepRecord:
    last_sleep_start:   float = field(default_factory=lambda: time.time() - 8 * 3600)
    last_sleep_end:     float = field(default_factory=time.time)
    last_sleep_quality: float = 1.0     # 0–1
    hours_awake:        float = 0.0
    total_sleep_h:      float = 0.0
    sleep_debt_h:       float = 0.0     # cumulative deficit
    stress_level:       float = 0.0     # 0–1

    @property
    def fatigue_level(self) -> FatigueLevel:
        if self.hours_awake < 20: return FatigueLevel.RESTED
        if self.hours_awake < 28: return FatigueLevel.TIRED
        if self.hours_awake < 36: return FatigueLevel.EXHAUSTED
        return FatigueLevel.DELIRIOUS

    @property
    def hallucination_probability(self) -> float:
        """Probability of hallucination event per hour."""
        if self.fatigue_level == FatigueLevel.DELIRIOUS:
            return float(min(0.8, (self.hours_awake - 36) / 24 * 0.8 + 0.2))
        if self.fatigue_level == FatigueLevel.EXHAUSTED:
            return 0.05
        return 0.0

    @property
    def observation_error_factor(self) -> float:
        """Multiplicative factor on signal detection probability."""
        if self.fatigue_level == FatigueLevel.RESTED:    return 1.00
        if self.fatigue_level == FatigueLevel.TIRED:     return 0.90
        if self.fatigue_level == FatigueLevel.EXHAUSTED: return 0.70
        return 0.45     # delirious: significant impairment

    def update(self, delta_h: float) -> None:
        self.hours_awake += delta_h
        self.sleep_debt_h = max(0, self.sleep_debt_h + delta_h - 8 * delta_h / 24)

    def sleep(self, hours: float) -> float:
        """Simulate a sleep period. Returns quality 0–1."""
        quality = float(np.clip(hours / 8.0, 0, 1))
        quality *= max(0.3, 1.0 - self.stress_level * 0.5)
        self.hours_awake = max(0.0, self.hours_awake - hours * (1 + quality))
        self.total_sleep_h += hours
        self.sleep_debt_h  = max(0, self.sleep_debt_h - hours * quality)
        self.last_sleep_quality = quality
        self.last_sleep_end  = time.time()
        self.last_sleep_start = time.time() - hours * 3600
        return quality


@dataclass
class StationLog:
    entries: List[Dict[str, Any]] = field(default_factory=list)
    max_entries: int = 500

    def log(self, severity: EventSeverity, category: str,
            message: str, ingame_day: int = 0, ingame_time: float = 0.0) -> None:
        entry = {
            "real_time":   pd.Timestamp(time.time(), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "game_day":    ingame_day,
            "game_time":   f"{int(ingame_time):02d}:{int((ingame_time % 1) * 60):02d}",
            "severity":    severity.name,
            "category":    category,
            "message":     message,
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["real_time","game_day","game_time",
                                          "severity","category","message"])
        return pd.DataFrame(self.entries)

    def recent(self, n: int = 20) -> pd.DataFrame:
        return self.to_dataframe().tail(n)


@dataclass
class RandomEvent:
    uid:         str  = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    timestamp:   float = field(default_factory=time.time)
    category:    str  = ""
    title:       str  = ""
    description: str  = ""
    severity:    EventSeverity = EventSeverity.INFO
    duration_h:  float = 0.0
    resolved:    bool  = False
    resolution:  str   = ""
    effects:     Dict[str, Any] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# ASTRONOMICAL CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class AstronomicalCalc:
    """
    Solar position calculations for day/night cycle.
    Implements simplified NOAA solar algorithm with declination and hour angle.
    """

    @staticmethod
    def solar_declination(day_of_year: int) -> float:
        """Solar declination in degrees."""
        b = 2 * math.pi * (day_of_year - 1) / 365
        return float(23.45 * math.sin(
            math.radians(360 / 365 * (284 + day_of_year))))

    @staticmethod
    def equation_of_time_min(day_of_year: int) -> float:
        """Equation of time in minutes."""
        b = 2 * math.pi * (day_of_year - 81) / 364
        return float(9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b))

    @staticmethod
    def solar_elevation(lat_deg: float, lon_deg: float,
                        hour_utc: float, day_of_year: int) -> float:
        """
        Solar elevation angle above horizon (degrees).
        Positive = above horizon.
        """
        declination = math.radians(AstronomicalCalc.solar_declination(day_of_year))
        lat = math.radians(lat_deg)
        eqt = AstronomicalCalc.equation_of_time_min(day_of_year)
        solar_noon = 12 - lon_deg / 15 - eqt / 60
        hour_angle = math.radians(15 * (hour_utc - solar_noon))
        sin_el = (math.sin(lat) * math.sin(declination) +
                  math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
        return float(math.degrees(math.asin(np.clip(sin_el, -1, 1))))

    @staticmethod
    def day_phase(elevation_deg: float) -> DayPhase:
        if elevation_deg > 0:
            return DayPhase.DAY
        elif elevation_deg > -0.5:
            return DayPhase.SUNRISE_SUNSET
        elif elevation_deg > -6:
            return DayPhase.CIVIL_TWILIGHT
        elif elevation_deg > -12:
            return DayPhase.NAUTICAL_TWIL
        elif elevation_deg > -18:
            return DayPhase.ASTRO_TWILIGHT
        return DayPhase.NIGHT

    @staticmethod
    def galactic_noise_k(elevation_deg: float, freq_mhz: float) -> float:
        """
        Galactic sky noise temperature as function of elevation and frequency.
        T_gal ≈ T_0 * (freq/408MHz)^(-2.75) * (1 + 0.3*|sin(b)|)^{-1}
        where b is galactic latitude (approximated from elevation).
        """
        t0_408 = 17.1      # K at 408 MHz
        exponent = -2.75
        t_gal = t0_408 * (freq_mhz / 408.0) ** exponent
        # Low elevation: more galactic plane (simplified)
        el_factor = 1.0 + 0.5 * max(0.0, 1.0 - abs(elevation_deg) / 90.0)
        return float(t_gal * el_factor)

    @staticmethod
    def ionospheric_scintillation_factor(elevation_deg: float,
                                          local_hour: float,
                                          day_of_year: int) -> float:
        """
        Ionospheric scintillation severity [0–1].
        Peak at dawn/dusk and at solar maximum seasons.
        """
        # Dawn/dusk enhancement
        dusk_dawn = float(np.exp(-0.5 * ((local_hour - 6) / 0.8) ** 2) +
                          np.exp(-0.5 * ((local_hour - 18) / 0.8) ** 2))
        # Seasonal: worse at equinoxes (day ~80, ~266)
        doy = day_of_year
        seasonal = float(np.exp(-0.5 * ((doy - 80) / 20) ** 2) +
                         np.exp(-0.5 * ((doy - 266) / 20) ** 2)) * 0.5
        # Low elevation is more affected
        el_factor = max(0.0, 1.0 - elevation_deg / 90.0)
        return float(np.clip((dusk_dawn + seasonal) * el_factor * 0.3, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER SIMULATOR — Markov chain with physics
# ─────────────────────────────────────────────────────────────────────────────

class WeatherSimulator:
    """
    5-state Markov chain weather model calibrated for Swiss alpine conditions.
    Each step represents 30 minutes of simulated time.
    Physical parameters are generated from state-conditional distributions.
    """

    # Transition matrix: rows = from, cols = to
    # CLEAR, PARTLY, OVERCAST, FOG, STORM
    TRANSITION = np.array([
        [0.70, 0.18, 0.07, 0.04, 0.01],   # from CLEAR
        [0.15, 0.60, 0.16, 0.05, 0.04],   # from PARTLY_CLOUDY
        [0.08, 0.20, 0.52, 0.10, 0.10],   # from OVERCAST
        [0.10, 0.15, 0.25, 0.45, 0.05],   # from FOG
        [0.02, 0.05, 0.15, 0.08, 0.70],   # from STORM
    ])
    STATES = list(WeatherState)

    # Seasonal temperature offsets (°C) — winter in Alps
    SEASON_TEMP = {
        "winter": -8.0, "spring": 2.0, "summer": 12.0, "autumn": 3.0
    }

    def __init__(self, seed: int = 42, season: str = "winter"):
        self.rng = np.random.default_rng(seed)
        self.season = season
        self.current_state = WeatherState.CLEAR
        self.history: List[WeatherRecord] = []
        self._base_temp = self.SEASON_TEMP.get(season, 0.0)

    def _step_state(self) -> WeatherState:
        """Advance one Markov step."""
        idx = self.STATES.index(self.current_state)
        probs = self.TRANSITION[idx]
        new_idx = int(self.rng.choice(len(self.STATES), p=probs))
        self.current_state = self.STATES[new_idx]
        return self.current_state

    def _generate_params(self, state: WeatherState,
                          hour_utc: float) -> WeatherRecord:
        """Generate physically consistent weather parameters for state."""
        rng = self.rng
        base_t = self._base_temp

        # Diurnal temperature variation ±3°C
        diurnal = 3.0 * math.sin(math.pi * (hour_utc - 6) / 12)

        params = {
            WeatherState.CLEAR: dict(
                temp_range=(base_t + diurnal - 2, base_t + diurnal + 2),
                humidity=(15, 35), wind=(0.5, 5.0), precip=0.0,
                cloud=(0, 10), pressure=(1013, 1025), vis=(50, 80)),
            WeatherState.PARTLY_CLOUDY: dict(
                temp_range=(base_t + diurnal - 1, base_t + diurnal + 1),
                humidity=(30, 55), wind=(1.0, 8.0), precip=0.0,
                cloud=(20, 60), pressure=(1005, 1018), vis=(30, 60)),
            WeatherState.OVERCAST: dict(
                temp_range=(base_t + diurnal - 3, base_t + diurnal + 1),
                humidity=(55, 80), wind=(2.0, 12.0), precip=rng.exponential(0.5),
                cloud=(70, 95), pressure=(995, 1010), vis=(10, 30)),
            WeatherState.FOG: dict(
                temp_range=(base_t - 2, base_t + 1),
                humidity=(90, 100), wind=(0.0, 2.0), precip=0.0,
                cloud=(80, 100), pressure=(1005, 1015), vis=(0.1, 1.0)),
            WeatherState.STORM: dict(
                temp_range=(base_t - 5, base_t + 2),
                humidity=(70, 100), wind=(10.0, 30.0), precip=rng.exponential(5.0),
                cloud=(90, 100), pressure=(980, 1000), vis=(0.5, 5.0)),
        }[state]

        temp = float(rng.uniform(*params["temp_range"]))
        humidity = float(rng.uniform(*params["humidity"]))
        wind = float(rng.uniform(*params["wind"]))
        wind_dir = float(rng.uniform(0, 360))
        precip = float(max(0, params["precip"] if isinstance(params["precip"], float)
                           else float(params["precip"])))
        cloud = float(rng.uniform(*params["cloud"]))
        pressure = float(rng.uniform(*params["pressure"]))
        vis = float(rng.uniform(*params["vis"]))
        lightning = (state == WeatherState.STORM and
                     rng.random() < 0.3 and
                     precip > 3.0)

        return WeatherRecord(
            timestamp=time.time(),
            state=state,
            temperature_c=temp,
            humidity_pct=humidity,
            wind_speed_ms=wind,
            wind_direction_deg=wind_dir,
            precipitation_mm_h=precip,
            cloud_cover_pct=cloud,
            pressure_hpa=pressure,
            visibility_km=vis,
            lightning_active=lightning,
        )

    def step(self, hour_utc: float) -> WeatherRecord:
        """Advance weather by one 30-minute step."""
        new_state = self._step_state()
        record = self._generate_params(new_state, hour_utc)
        self.history.append(record)
        if len(self.history) > 1000:
            self.history.pop(0)
        return record

    def simulate_n_steps(self, n: int, start_hour: float = 0.0) -> List[WeatherRecord]:
        records = []
        for i in range(n):
            hour = (start_hour + i * 0.5) % 24
            records.append(self.step(hour))
        return records

    def forecast_severity(self, n_steps: int = 24) -> Dict[str, float]:
        """Monte Carlo forecast: probability of each state over next n_steps."""
        n_mc = 500
        counts = defaultdict(int)
        orig_state = self.current_state
        rng_saved = np.random.default_rng(0)
        for _ in range(n_mc):
            temp_state = orig_state
            for _ in range(n_steps):
                idx = self.STATES.index(temp_state)
                probs = self.TRANSITION[idx]
                temp_state = self.STATES[int(rng_saved.choice(len(self.STATES), p=probs))]
            counts[temp_state.value] += 1
        self.current_state = orig_state
        return {k: v / n_mc for k, v in counts.items()}

    def weather_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([{
            "State":      r.state.value,
            "Temp(°C)":   round(r.temperature_c, 1),
            "Humidity%":  round(r.humidity_pct, 1),
            "Wind(m/s)":  round(r.wind_speed_ms, 1),
            "Precip":     round(r.precipitation_mm_h, 2),
            "Cloud%":     round(r.cloud_cover_pct, 1),
            "Press(hPa)": round(r.pressure_hpa, 1),
            "Vis(km)":    round(r.visibility_km, 2),
            "RadioQ":     round(r.radio_seeing_factor, 3),
            "AtmT(K)":    round(r.atmospheric_noise_k, 2),
            "Lightning":  "⚡" if r.lightning_active else "—",
        } for r in self.history[-48:]])


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM EVENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RandomEventEngine:
    """
    Generates random events mirroring VotV's event system.
    Events are drawn from category pools with conditional probabilities
    that depend on game state (time, reputation, equipment health, sleep).
    """

    EVENT_CATALOG: Dict[str, List[Dict[str, Any]]] = {
        "equipment": [
            {
                "title":       "Dish Motor Fault",
                "description": "Azimuth motor reports abnormal current draw. Immediate inspection recommended.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  2.0,
                "effects":     {"az_motor_health": -0.3},
                "resolution":  "Apply lubricant and re-seat motor connections.",
            },
            {
                "title":       "Feed Horn Misalignment",
                "description": "Signal strength dropped 3dB. Feed horn may have shifted due to wind.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  1.0,
                "effects":     {"feed_alignment_deg": +2.5},
                "resolution":  "Re-align feed horn using boresight procedure.",
            },
            {
                "title":       "LNA Temperature Anomaly",
                "description": "LNA cryostat temperature spiking. Noise floor elevated by 8K.",
                "severity":    EventSeverity.CRITICAL,
                "duration_h":  4.0,
                "effects":     {"lna_noise_figure_db": +1.5},
                "resolution":  "Allow LNA to re-cool. Check cryocooler compressor.",
            },
            {
                "title":       "RFI Spike — Schwarztal",
                "description": "Strong narrowband interference from village direction. Multiple channels contaminated.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.5,
                "effects":     {"rfi_environment": "elevated"},
                "resolution":  "Wait for source to cease or apply notch filter.",
            },
            {
                "title":       "Cable Connector Corrosion",
                "description": "VSWR increased to 1.8. Likely oxidised N-type connector at dish base.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.5,
                "effects":     {"connector_vswr": +0.75},
                "resolution":  "Clean connector with isopropanol and re-torque.",
            },
            {
                "title":       "Filter Module Saturated",
                "description": "Band-pass filter reporting compression. Strong nearby transmitter suspected.",
                "severity":    EventSeverity.CRITICAL,
                "duration_h":  1.0,
                "effects":     {"lna_compression_dbm": -3},
                "resolution":  "Reduce pre-amplifier gain or replace filter module.",
            },
        ],
        "power": [
            {
                "title":       "Generator Fuel Warning",
                "description": "Fuel level critical. Less than 5 litres remaining.",
                "severity":    EventSeverity.CRITICAL,
                "duration_h":  0.0,
                "effects":     {},
                "resolution":  "Refuel immediately from supply drums in storage shed.",
            },
            {
                "title":       "Power Brownout",
                "description": "Mains voltage dropped to 195V. Observatory computers restarting.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.25,
                "effects":     {"data_loss_risk": True},
                "resolution":  "Switch to battery backup. Check generator load.",
            },
            {
                "title":       "Battery Bank Cell Failure",
                "description": "Cell 3 of battery bank reporting 0V. Capacity reduced by 15%.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.0,
                "effects":     {"battery_capacity_pct": -15},
                "resolution":  "Mark failed cell. Order replacement. Reduce discretionary load.",
            },
            {
                "title":       "Lightning Strike Near Tower",
                "description": "Direct strike within 50 metres. Surge protection triggered. Check all systems.",
                "severity":    EventSeverity.CRITICAL,
                "duration_h":  1.0,
                "effects":     {"all_systems_check": True, "lna_noise_figure_db": +0.5},
                "resolution":  "Systematic continuity check on all coax runs and electronics.",
            },
        ],
        "environmental": [
            {
                "title":       "Snow Accumulation on Dish",
                "description": "Temperature dropped below -8°C. 15cm snow accumulation expected by morning.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  6.0,
                "effects":     {"surface_contamination": +0.4},
                "resolution":  "Tilt dish to 75° elevation and run de-icing protocol.",
            },
            {
                "title":       "Fog Bank Descending",
                "description": "Dense fog reducing visibility to <100m. Multipath interference expected.",
                "severity":    EventSeverity.INFO,
                "duration_h":  3.0,
                "effects":     {"weather_state": "FOG"},
                "resolution":  "Continue observations with reduced sensitivity expectation.",
            },
            {
                "title":       "Auroral Activity — Geomagnetic Storm",
                "description": "Kp=7 geomagnetic storm in progress. HF blackout. L-band mildly affected.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  8.0,
                "effects":     {"ionospheric_scintillation": +0.6},
                "resolution":  "Monitor Kp index. Defer sensitive observations until Kp < 4.",
            },
            {
                "title":       "Animal on Dish Structure",
                "description": "Motion sensor triggered. Large animal detected on dish support structure.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.25,
                "effects":     {"structural_check": True},
                "resolution":  "Wait for animal to leave. Check for structural damage.",
            },
        ],
        "anomalous": [
            {
                "title":       "03:33 Event — Unscheduled Signal",
                "description": "Strong narrowband signal at 03:33 local time. Source unresolved. No satellite match.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.05,
                "effects":     {"signal_detected": True, "wow_bonus": +2.0},
                "resolution":  "Log signal. Flag for ARIRAL assessment. Check SETI candidate database.",
            },
            {
                "title":       "ARIRAL Package Delivered",
                "description": "Small sealed container found at base of access ladder. Contents: shrimp pack.",
                "severity":    EventSeverity.INFO,
                "duration_h":  0.0,
                "effects":     {"shrimp_packs": +1, "rep_delta": +5},
                "resolution":  "Accept. Log coordinates. Do not broadcast location.",
            },
            {
                "title":       "Visitor at Gate — Unidentified",
                "description": "Motion sensor at perimeter gate. No scheduled supply delivery today.",
                "severity":    EventSeverity.CRITICAL,
                "duration_h":  0.5,
                "effects":     {"stress_delta": +0.2},
                "resolution":  "Do not open gate. Monitor cameras. Report if visitor returns.",
            },
            {
                "title":       "The Looker Event — Phase 1",
                "description": "Sequential prime-encoded signals received over 17 minutes. Pattern matches Looker sequence.",
                "severity":    EventSeverity.HORROR,
                "duration_h":  0.3,
                "effects":     {"entity_event": "LOOKER", "rep_delta": -5},
                "resolution":  "Complete all 7 sequence signals. Do NOT miss any.",
            },
            {
                "title":       "Void Fragment — CONTAINMENT REQUIRED",
                "description": "Physical artefact detected near array. DO NOT TOUCH. Call HQ immediately.",
                "severity":    EventSeverity.HORROR,
                "duration_h":  0.0,
                "effects":     {"void_fragments": +1, "rep_delta": -20},
                "resolution":  "ERASE ALL DRIVES. EVACUATE. DO NOT ATTEMPT ANALYSIS.",
            },
        ],
        "personal": [
            {
                "title":       "Supply Drop Received",
                "description": "Monthly supply helicopter delivered food rations, drives, and maintenance kit.",
                "severity":    EventSeverity.INFO,
                "duration_h":  0.0,
                "effects":     {"food_rations": +14, "blank_drives": +4, "cleaning_kits": +2},
                "resolution":  "Inventory updated. Sign delivery receipt.",
            },
            {
                "title":       "HQ Routine Check-in",
                "description": "Scheduled call from headquarters. Signal quality report and drive submission requested.",
                "severity":    EventSeverity.INFO,
                "duration_h":  0.25,
                "effects":     {},
                "resolution":  "Submit weekly report and sealed drives via satellite uplink.",
            },
            {
                "title":       "Sleep Disruption",
                "description": "Recurring alarm malfunction. Lost 3h of scheduled rest.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.0,
                "effects":     {"hours_awake_penalty": +3},
                "resolution":  "Repair alarm. Consume coffee if needed to continue observations.",
            },
            {
                "title":       "Anomalous Notes Received",
                "description": "Envelope slipped under door. Contains handwritten radio frequency coordinates. Sender unknown.",
                "severity":    EventSeverity.WARNING,
                "duration_h":  0.0,
                "effects":     {"anomalous_notes": +1},
                "resolution":  "Investigate frequencies listed. Do not reply to sender.",
            },
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed or int(time.time()) % (2**31))
        self._event_history: List[RandomEvent] = []

    def _base_rate(self, category: str, state: Dict[str, Any]) -> float:
        """Hourly event rate for category given current game state."""
        rates = {
            "equipment":   0.05,
            "power":       0.03,
            "environmental": 0.08,
            "anomalous":   0.02,
            "personal":    0.04,
        }
        rate = rates.get(category, 0.03)

        # Modifiers
        dish_health   = state.get("dish_health_fraction", 1.0)
        fuel_level    = state.get("generator_fuel_l", 50.0)
        hours_awake   = state.get("hours_awake", 8.0)
        weather_state = state.get("weather_state", "CLEAR")
        ingame_hour   = state.get("ingame_hour", 12.0)

        if category == "equipment":
            rate *= (1.0 + (1.0 - dish_health))
            if weather_state in ("STORM", "FOG"):
                rate *= 2.0
        elif category == "power":
            if fuel_level < 10:
                rate *= 3.0
        elif category == "anomalous":
            # 03:33 window has elevated anomaly rate
            if abs(ingame_hour - 3.55) < 0.3:
                rate *= 5.0
            if hours_awake > 28:
                rate *= 1.5   # exhausted observers notice more
        elif category == "personal":
            if hours_awake > 20:
                rate *= 1.3

        return float(rate)

    def sample(self, delta_h: float, game_state: Dict[str, Any]
               ) -> List[RandomEvent]:
        """
        Sample events for a delta_h period.
        Returns list of triggered events (may be empty).
        """
        triggered: List[RandomEvent] = []
        for category, events in self.EVENT_CATALOG.items():
            rate = self._base_rate(category, game_state)
            # Poisson number of events in interval
            expected = rate * delta_h
            n_events = int(self.rng.poisson(expected))
            if n_events == 0:
                continue
            chosen = [self.EVENT_CATALOG[category][
                int(self.rng.integers(0, len(self.EVENT_CATALOG[category])))]
                for _ in range(n_events)]
            for ev_data in chosen:
                event = RandomEvent(
                    category=category,
                    title=ev_data["title"],
                    description=ev_data["description"],
                    severity=ev_data["severity"],
                    duration_h=ev_data.get("duration_h", 0.0),
                    effects=ev_data.get("effects", {}),
                    resolution=ev_data.get("resolution", ""),
                )
                triggered.append(event)
                self._event_history.append(event)
        return triggered

    def event_history_df(self) -> pd.DataFrame:
        if not self._event_history:
            return pd.DataFrame(columns=["Time","Category","Title","Severity","Resolved"])
        rows = []
        for ev in self._event_history[-50:]:
            rows.append({
                "Time":     pd.Timestamp(ev.timestamp, unit="s").strftime("%H:%M:%S"),
                "Category": ev.category,
                "Title":    ev.title[:40],
                "Severity": ev.severity.name,
                "Duration": f"{ev.duration_h:.1f}h",
                "Resolved": "✓" if ev.resolved else "—",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUALITY CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class SignalQualityCalculator:
    """
    Computes the composite signal quality factor from all station subsystems.
    This directly impacts SNR of received signals.
    """

    def compute(self, dish: DishHealth, weather: WeatherRecord,
                power: PowerSystem, sleep: SleepRecord,
                elevation_deg: float, freq_mhz: float,
                day_of_year: int, ingame_hour: float
                ) -> Dict[str, float]:
        """
        Returns dict of individual quality factors and composite.
        All factors are [0, 1] where 1 = perfect.
        """
        # 1. Aperture efficiency from dish health
        aperture_eff = dish.effective_area_fraction

        # 2. Weather radio seeing
        radio_seeing = weather.radio_seeing_factor

        # 3. Power quality (reduced quality if on battery with low charge)
        if power.active_source == PowerSource.OFFLINE:
            power_q = 0.0
        elif power.active_source == PowerSource.BATTERY and power.battery_fraction < 0.3:
            power_q = float(0.5 + 0.5 * power.battery_fraction / 0.3)
        else:
            power_q = 1.0

        # 4. Ionospheric scintillation
        iono_scint = AstronomicalCalc.ionospheric_scintillation_factor(
            elevation_deg, ingame_hour, day_of_year)
        iono_q = float(1.0 - iono_scint * 0.5)

        # 5. Galactic noise impact on SNR
        t_gal = AstronomicalCalc.galactic_noise_k(elevation_deg, freq_mhz)
        t_atm = weather.atmospheric_noise_k
        t_sys_eff = SYSTEM_TSYS_K + dish.system_temp_penalty_k + t_gal + t_atm
        # Normalise: lower T_sys is better
        tsys_q = float(np.clip(1.0 - (t_sys_eff - SYSTEM_TSYS_K) / 200.0, 0.2, 1.0))

        # 6. Observer fatigue (affects detection of weak signals)
        observer_q = sleep.observation_error_factor

        # 7. RFI environment (simplistic: night = less RFI from Schwarztal)
        rfi_q = 0.7 if 9 <= ingame_hour <= 22 else 1.0

        # Composite: geometric mean (all factors must be good)
        composite = float((aperture_eff * radio_seeing * power_q *
                            iono_q * tsys_q * observer_q * rfi_q) ** (1 / 7))

        # SNR penalty in dB
        snr_penalty_db = float(-10 * math.log10(max(composite, 0.01)))

        return {
            "aperture_efficiency": aperture_eff,
            "radio_seeing":        radio_seeing,
            "power_quality":       power_q,
            "ionospheric_quality": iono_q,
            "tsys_quality":        tsys_q,
            "observer_quality":    observer_q,
            "rfi_quality":         rfi_q,
            "composite":           composite,
            "snr_penalty_db":      snr_penalty_db,
            "t_sys_effective_k":   t_sys_eff,
            "t_gal_k":             t_gal,
            "t_atm_k":             t_atm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ─────────────────────────────────────────────────────────────────────────────

class EnvironmentVisualizer:
    PAL = {
        "bg":   "#030d07", "fg":   "#00ff88", "dim":  "#004422",
        "acc":  "#00ffcc", "warn": "#ff8800", "dng":  "#ff2222",
        "grid": "#001a0e", "axis": "#335544", "blue": "#0088ff",
        "purp": "#8844ff", "gold": "#ffcc00",
    }

    STATE_COLORS = {
        "CLEAR":        "#00ff88",
        "PARTLY_CLOUDY":"#88ffcc",
        "OVERCAST":     "#ff8800",
        "FOG":          "#8888ff",
        "STORM":        "#ff2222",
    }

    def _style(self, ax, title=""):
        ax.set_facecolor(self.PAL["bg"])
        for sp in ax.spines.values(): sp.set_edgecolor(self.PAL["dim"])
        ax.tick_params(colors=self.PAL["axis"], labelsize=6)
        ax.grid(True, color=self.PAL["grid"], alpha=0.4, ls=":")
        if title:
            ax.set_title(title, color=self.PAL["fg"], fontsize=7.5,
                         loc="left", pad=3, fontfamily="monospace")

    def plot_station_dashboard(self, dish: DishHealth, power: PowerSystem,
                                sleep: SleepRecord, weather: WeatherRecord,
                                quality: Dict[str, float],
                                figsize=(12, 7)) -> plt.Figure:
        fig = plt.figure(figsize=figsize, facecolor=self.PAL["bg"])
        gs  = gridspec.GridSpec(3, 4, figure=fig,
                                 hspace=0.55, wspace=0.45)

        # ── 1. Dish health gauge ───────────────────────────────────────────
        ax_dish = fig.add_subplot(gs[0, 0])
        eff = dish.effective_area_fraction
        col = self.PAL["fg"] if eff > 0.8 else self.PAL["warn"] if eff > 0.5 else self.PAL["dng"]
        theta = np.linspace(0, np.pi, 200)
        r_o, r_i = 1.0, 0.55
        zones = [(0.0, 0.5, self.PAL["dng"]),
                 (0.5, 0.8, self.PAL["warn"]),
                 (0.8, 1.0, self.PAL["fg"])]
        for lo, hi, zc in zones:
            t_lo, t_hi = np.pi * (1 - hi), np.pi * (1 - lo)
            t_arc = np.linspace(t_lo, t_hi, 40)
            xo, yo = r_o * np.cos(t_arc), r_o * np.sin(t_arc)
            xi, yi = r_i * np.cos(t_arc[::-1]), r_i * np.sin(t_arc[::-1])
            ax_dish.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]),
                          color=zc, alpha=0.5)
        angle = np.pi * (1 - eff)
        ax_dish.annotate("", xy=(0.8 * np.cos(angle), 0.8 * np.sin(angle)),
                          xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
        ax_dish.set_xlim(-1.1, 1.1); ax_dish.set_ylim(-0.2, 1.1)
        ax_dish.set_aspect("equal"); ax_dish.axis("off")
        ax_dish.set_facecolor(self.PAL["bg"])
        ax_dish.text(0, -0.15, f"{eff*100:.0f}%", ha="center", va="top",
                      color=col, fontsize=10, fontfamily="monospace", fontweight="bold")
        ax_dish.set_title("DISH EFF.", color=self.PAL["fg"], fontsize=7, loc="center",
                           fontfamily="monospace")

        # ── 2. Power system ────────────────────────────────────────────────
        ax_pwr = fig.add_subplot(gs[0, 1])
        batt = power.battery_fraction
        fuel = power.generator_fuel_l / GENERATOR_CAPACITY_L
        solar = min(1.0, power.solar_power_kw / (BASE_LOAD_KW + 0.1))
        labels = ["Battery", "Fuel", "Solar"]
        vals   = [batt, fuel, solar]
        cols   = [
            self.PAL["fg"] if batt > 0.4 else self.PAL["warn"] if batt > 0.15 else self.PAL["dng"],
            self.PAL["fg"] if fuel > 0.3 else self.PAL["warn"] if fuel > 0.1  else self.PAL["dng"],
            self.PAL["blue"],
        ]
        bars = ax_pwr.barh(labels, vals, color=cols, alpha=0.82, height=0.5)
        ax_pwr.set_xlim(0, 1)
        for bar, val in zip(bars, vals):
            ax_pwr.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                         f"{val*100:.0f}%", va="center", fontsize=6,
                         color=self.PAL["acc"], fontfamily="monospace")
        source_text = f"SOURCE: {power.active_source.value}"
        ax_pwr.text(0.5, -0.18, source_text, transform=ax_pwr.transAxes,
                     ha="center", color=self.PAL["acc"], fontsize=5.5,
                     fontfamily="monospace")
        self._style(ax_pwr, "POWER SYSTEM")
        ax_pwr.set_xlabel("Fraction", fontsize=6, color=self.PAL["fg"])

        # ── 3. Sleep / fatigue ─────────────────────────────────────────────
        ax_sleep = fig.add_subplot(gs[0, 2])
        awake_h = sleep.hours_awake
        debt_h  = sleep.sleep_debt_h
        fatigue_col = {
            FatigueLevel.RESTED:    self.PAL["fg"],
            FatigueLevel.TIRED:     self.PAL["warn"],
            FatigueLevel.EXHAUSTED: "#ff4400",
            FatigueLevel.DELIRIOUS: self.PAL["dng"],
        }[sleep.fatigue_level]
        ax_sleep.barh(["Awake(h)", "Debt(h)"], [min(awake_h, 48), min(debt_h, 24)],
                       color=[fatigue_col, self.PAL["warn"]], alpha=0.82, height=0.5)
        ax_sleep.set_xlim(0, 48)
        ax_sleep.axvline(20, color=self.PAL["warn"], lw=0.7, ls="--", alpha=0.6)
        ax_sleep.axvline(36, color=self.PAL["dng"],  lw=0.7, ls="--", alpha=0.6)
        ax_sleep.text(0.5, -0.18,
                       f"{sleep.fatigue_level.value}  HALL:{sleep.hallucination_probability:.0%}",
                       transform=ax_sleep.transAxes,
                       ha="center", color=fatigue_col,
                       fontsize=5.5, fontfamily="monospace")
        self._style(ax_sleep, "FATIGUE")
        ax_sleep.set_xlabel("Hours", fontsize=6, color=self.PAL["fg"])

        # ── 4. Weather ─────────────────────────────────────────────────────
        ax_wx = fig.add_subplot(gs[0, 3])
        ax_wx.axis("off"); ax_wx.set_facecolor(self.PAL["bg"])
        wx_col = self.STATE_COLORS.get(weather.state.value, self.PAL["fg"])
        wx_lines = [
            ("STATE",    weather.state.value),
            ("TEMP",     f"{weather.temperature_c:.1f}°C"),
            ("HUMIDITY", f"{weather.humidity_pct:.0f}%"),
            ("WIND",     f"{weather.wind_speed_ms:.1f} m/s"),
            ("PRECIP",   f"{weather.precipitation_mm_h:.1f} mm/h"),
            ("CLOUD",    f"{weather.cloud_cover_pct:.0f}%"),
            ("PRESSURE", f"{weather.pressure_hpa:.0f} hPa"),
            ("VIS",      f"{weather.visibility_km:.1f} km"),
            ("RADIO-Q",  f"{weather.radio_seeing_factor:.2f}"),
            ("⚡",       "ACTIVE" if weather.lightning_active else "—"),
        ]
        for i, (k, v) in enumerate(wx_lines):
            col = wx_col if i == 0 else self.PAL["dng"] if v == "ACTIVE" else self.PAL["fg"]
            ax_wx.text(0.02, 1.0 - i * 0.1, f"{k:<8} {v}",
                        transform=ax_wx.transAxes, color=col,
                        fontsize=6.5, fontfamily="monospace", va="top")
        ax_wx.set_title("WEATHER", color=self.PAL["fg"], fontsize=7.5,
                         loc="left", fontfamily="monospace", pad=3)

        # ── 5. Signal quality spider ───────────────────────────────────────
        ax_spider = fig.add_subplot(gs[1, :2], polar=True)
        ax_spider.set_facecolor(self.PAL["bg"])
        q_keys = ["aperture_efficiency", "radio_seeing", "power_quality",
                   "ionospheric_quality", "tsys_quality", "observer_quality", "rfi_quality"]
        q_labels = ["Aperture", "Radio See", "Power", "Iono", "Tsys", "Observer", "RFI"]
        vals_q = [quality.get(k, 0.0) for k in q_keys]
        n = len(q_keys)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vq = vals_q + [vals_q[0]]
        ang = angles + [angles[0]]
        ax_spider.plot(ang, vq, color=self.PAL["fg"], lw=1.2)
        ax_spider.fill(ang, vq, color=self.PAL["fg"], alpha=0.15)
        ax_spider.set_xticks(angles)
        ax_spider.set_xticklabels(q_labels, size=6, color=self.PAL["acc"])
        ax_spider.set_ylim(0, 1)
        ax_spider.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_spider.set_yticklabels(["25%","50%","75%","100%"], size=5, color=self.PAL["dim"])
        ax_spider.tick_params(colors=self.PAL["dim"])
        ax_spider.spines["polar"].set_edgecolor(self.PAL["dim"])
        ax_spider.grid(color=self.PAL["grid"], alpha=0.5)
        ax_spider.set_title(
            f"COMPOSITE Q={quality.get('composite', 0):.2f} "
            f"SNR PEN={quality.get('snr_penalty_db', 0):.1f}dB",
            color=self.PAL["acc"], fontsize=7, pad=8, fontfamily="monospace")

        # ── 6. Dish subsystem bars ─────────────────────────────────────────
        ax_dish_detail = fig.add_subplot(gs[1, 2:])
        d_labels = ["AZ Motor", "EL Motor", "Surface Refl",
                     "Feed Match", "LNA NF (inv)", "Cable Loss (inv)"]
        d_vals = [
            dish.az_motor_health,
            dish.el_motor_health,
            dish.surface_reflectivity * (1 - dish.surface_contamination),
            dish.feed_impedance_match,
            float(np.clip(1 - dish.lna_noise_figure_db / 5, 0, 1)),
            float(np.clip(1 - dish.cable_loss_db / 3, 0, 1)),
        ]
        d_cols = [
            self.PAL["fg"] if v > 0.75 else self.PAL["warn"] if v > 0.5 else self.PAL["dng"]
            for v in d_vals
        ]
        bars_d = ax_dish_detail.barh(d_labels[::-1], d_vals[::-1],
                                      color=d_cols[::-1], alpha=0.82, height=0.55)
        ax_dish_detail.set_xlim(0, 1)
        for bar, val in zip(bars_d, d_vals[::-1]):
            ax_dish_detail.text(min(val + 0.01, 0.97),
                                 bar.get_y() + bar.get_height() / 2,
                                 f"{val:.2f}", va="center", fontsize=5.5,
                                 color=self.PAL["acc"], fontfamily="monospace")
        ax_dish_detail.axvline(0.75, color=self.PAL["warn"], lw=0.6, ls=":", alpha=0.6)
        ax_dish_detail.axvline(0.50, color=self.PAL["dng"],  lw=0.6, ls=":", alpha=0.6)
        self._style(ax_dish_detail, "DISH SUBSYSTEM HEALTH")
        ax_dish_detail.set_xlabel("Health (0–1)", fontsize=6, color=self.PAL["fg"])

        # ── 7. Weather history ─────────────────────────────────────────────
        ax_wx_hist = fig.add_subplot(gs[2, :])
        # Placeholder timeline — would use weather history if available
        t_range = np.linspace(0, 24, 200)
        radio_q_curve = np.clip(
            1.0 - 0.3 * np.sin(t_range * 0.5 + 1) * np.cos(t_range * 0.2),
            0.2, 1.0)
        ax_wx_hist.plot(t_range, radio_q_curve, color=self.PAL["fg"], lw=0.9)
        ax_wx_hist.fill_between(t_range, radio_q_curve, 0,
                                 alpha=0.12, color=self.PAL["fg"])
        ax_wx_hist.axhline(0.7, color=self.PAL["warn"], lw=0.6, ls="--", alpha=0.6)
        ax_wx_hist.axhline(0.5, color=self.PAL["dng"],  lw=0.6, ls="--", alpha=0.5)
        # Day/night shading
        for h_start, h_end in [(0, 6.5), (21, 24)]:
            ax_wx_hist.axvspan(h_start, h_end, alpha=0.08, color=self.PAL["purp"])
        ax_wx_hist.set_ylim(0, 1.05)
        ax_wx_hist.set_xlim(0, 24)
        ax_wx_hist.set_xlabel("Hour (UTC)", fontsize=6, color=self.PAL["fg"])
        ax_wx_hist.set_ylabel("Radio Quality", fontsize=6, color=self.PAL["fg"])
        self._style(ax_wx_hist, "RADIO QUALITY — 24H FORECAST")

        plt.suptitle("STATION TELEMETRY — DUNKELTALER OBSERVATORY",
                      color=self.PAL["fg"], fontsize=9, fontfamily="monospace",
                      y=1.00)
        plt.tight_layout(pad=0.5)
        return fig

    def plot_weather_history(self, weather_df: pd.DataFrame,
                              figsize=(10, 4)) -> plt.Figure:
        if weather_df.empty:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.PAL["bg"])
            ax.text(0.5, 0.5, "NO WEATHER DATA", transform=ax.transAxes,
                    ha="center", color=self.PAL["dim"], fontsize=10)
            ax.set_facecolor(self.PAL["bg"])
            return fig

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        facecolor=self.PAL["bg"], sharex=True)
        x = np.arange(len(weather_df))
        radio_q = weather_df["RadioQ"].values
        temps   = weather_df["Temp(°C)"].values
        wind    = weather_df["Wind(m/s)"].values

        ax1.plot(x, radio_q, color=self.PAL["fg"], lw=0.9, label="Radio Quality")
        ax1.fill_between(x, radio_q, 0, alpha=0.12, color=self.PAL["fg"])
        ax1.set_ylim(0, 1.05)
        self._style(ax1, "RADIO SEEING QUALITY")
        ax1.set_ylabel("Quality", fontsize=6, color=self.PAL["fg"])

        ax2.plot(x, temps, color=self.PAL["acc"], lw=0.9, label="Temp °C")
        ax2.plot(x, wind,  color=self.PAL["warn"], lw=0.8, ls="--", label="Wind m/s")
        ax2.axhline(0, color=self.PAL["dng"], lw=0.6, ls=":", alpha=0.5)
        self._style(ax2, "TEMPERATURE & WIND")
        ax2.set_xlabel("Time Step (30 min)", fontsize=6, color=self.PAL["fg"])
        ax2.set_ylabel("Value", fontsize=6, color=self.PAL["fg"])
        ax2.legend(fontsize=6, facecolor=self.PAL["bg"],
                    edgecolor=self.PAL["dim"], labelcolor=self.PAL["fg"])

        fig.set_facecolor(self.PAL["bg"])
        plt.tight_layout(pad=0.4)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def environment_monitor_page():
    from signal_engine import init_session_state

    init_session_state()

    st.markdown("""
    <style>
    .env-header {
        font-family:'Courier New',monospace;
        color:#00ff88;
        font-size:0.78rem;
        letter-spacing:0.14em;
        border-bottom:1px solid #00ff4430;
        padding-bottom:0.4rem;
        margin-bottom:1rem;
    }
    .env-label {
        font-family:'Courier New',monospace;
        color:#88ffcc;
        font-size:0.70rem;
        letter-spacing:0.09em;
        margin-top:0.5rem;
        margin-bottom:0.2rem;
    }
    .event-critical {
        background:#1a0500;border:1px solid #ff4400;
        padding:0.4rem 0.7rem;border-radius:2px;
        font-family:'Courier New',monospace;font-size:0.70rem;color:#ff6622;
        margin-bottom:0.3rem;
    }
    .event-horror {
        background:#1a0000;border:2px solid #ff0000;
        padding:0.4rem 0.7rem;border-radius:2px;
        font-family:'Courier New',monospace;font-size:0.72rem;color:#ff2222;
        margin-bottom:0.3rem;animation:blink 0.9s linear infinite;
    }
    .event-warning {
        background:#1a0e00;border:1px solid #ff8800;
        padding:0.4rem 0.7rem;border-radius:2px;
        font-family:'Courier New',monospace;font-size:0.70rem;color:#ffaa44;
        margin-bottom:0.3rem;
    }
    .event-info {
        background:#001a0e;border:1px solid #00ff4440;
        padding:0.4rem 0.7rem;border-radius:2px;
        font-family:'Courier New',monospace;font-size:0.68rem;color:#88ffcc;
        margin-bottom:0.3rem;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
    .shortage-badge {
        display:inline-block;
        background:#1a0500;border:1px solid #ff4400;
        border-radius:1px;padding:0.1rem 0.4rem;
        font-family:'Courier New',monospace;font-size:0.65rem;
        color:#ff6622;margin:0.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="env-header">[ OBSERVATORY ENVIRONMENT & SYSTEMS MONITOR — STATION TELEMETRY ]</div>',
                unsafe_allow_html=True)

    # ── Session init ───────────────────────────────────────────────────────
    if "dish" not in st.session_state:
        st.session_state.dish = DishHealth()
    if "power_system" not in st.session_state:
        st.session_state.power_system = PowerSystem()
    if "sleep_rec" not in st.session_state:
        st.session_state.sleep_rec = SleepRecord()
    if "inventory" not in st.session_state:
        st.session_state.inventory = Inventory()
    if "station_log" not in st.session_state:
        st.session_state.station_log = StationLog()
    if "weather_sim" not in st.session_state:
        st.session_state.weather_sim = WeatherSimulator(seed=42, season="winter")
    if "current_weather" not in st.session_state:
        st.session_state.current_weather = st.session_state.weather_sim.step(12.0)
    if "event_engine" not in st.session_state:
        st.session_state.event_engine = RandomEventEngine()
    if "env_viz" not in st.session_state:
        st.session_state.env_viz = EnvironmentVisualizer()
    if "sq_calc" not in st.session_state:
        st.session_state.sq_calc = SignalQualityCalculator()
    if "astro_calc" not in st.session_state:
        st.session_state.astro_calc = AstronomicalCalc()

    dish:    DishHealth         = st.session_state.dish
    power:   PowerSystem        = st.session_state.power_system
    sleep_r: SleepRecord        = st.session_state.sleep_rec
    inv:     Inventory          = st.session_state.inventory
    log:     StationLog         = st.session_state.station_log
    wx_sim:  WeatherSimulator   = st.session_state.weather_sim
    weather: WeatherRecord      = st.session_state.current_weather
    events:  RandomEventEngine  = st.session_state.event_engine
    viz:     EnvironmentVisualizer = st.session_state.env_viz
    sq:      SignalQualityCalculator = st.session_state.sq_calc

    col_ctrl, col_main = st.columns([1, 2.8])

    with col_ctrl:
        st.markdown('<div class="env-label">— SIMULATION ADVANCE —</div>', unsafe_allow_html=True)
        delta_h      = st.slider("Advance Time (h)", 0.1, 8.0, 1.0, 0.1)
        ingame_h     = st.slider("Current Hour (UTC)", 0.0, 23.99,
                                  float(st.session_state.get("env_hour", 12.0)), 0.25)
        st.session_state.env_hour = ingame_h
        day_of_year  = st.number_input("Day of Year", 1, 365,
                                        int(st.session_state.get("env_doy", 355)), 1)
        st.session_state.env_doy = day_of_year
        freq_mhz     = st.number_input("Obs. Freq (MHz)", 100.0, 30000.0, 1420.406,
                                        step=1.0, format="%.3f")
        el_deg       = st.slider("Dish Elevation (°)", 5.0, 90.0, 45.0, 1.0)

        st.markdown('<div class="env-label">— MAINTENANCE —</div>', unsafe_allow_html=True)
        maint_type   = st.selectbox("Maintenance Type",
                                     [m.value for m in MaintenanceType])
        maint_qual   = st.slider("Technician Quality", 0.3, 1.0, 0.8, 0.05)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("⬡ PERFORM"):
                result_msg = dish.repair(MaintenanceType(maint_type), maint_qual)
                log.log(EventSeverity.INFO, "MAINTENANCE", result_msg,
                        int(st.session_state.day), ingame_h)
                st.success(result_msg[:60])
        with col_m2:
            if st.button("⬡ REFUEL"):
                msg = power.refuel(20.0)
                log.log(EventSeverity.INFO, "POWER", msg, int(st.session_state.day), ingame_h)
                st.success(msg)

        st.markdown('<div class="env-label">— SLEEP & FATIGUE —</div>', unsafe_allow_html=True)
        sleep_h = st.slider("Sleep Duration (h)", 1.0, 12.0, 8.0, 0.5)
        if st.button("⬡ SLEEP NOW"):
            quality = sleep_r.sleep(sleep_h)
            inv.consume("coffee_packs", 0)
            log.log(EventSeverity.INFO, "SLEEP",
                     f"Slept {sleep_h:.1f}h. Quality={quality:.2f}. Debt={sleep_r.sleep_debt_h:.1f}h",
                     int(st.session_state.day), ingame_h)
            st.success(f"Slept {sleep_h:.1f}h | Quality: {quality:.2f} | Fatigue: {sleep_r.fatigue_level.value}")

        st.markdown('<div class="env-label">— INVENTORY QUICK-USE —</div>', unsafe_allow_html=True)
        for item_name, label in [("coffee_packs","COFFEE"),("food_rations","FOOD"),
                                   ("painkillers","PAINKILLER")]:
            if st.button(f"⬡ USE {label}"):
                ok = inv.consume(item_name, 1)
                if ok:
                    if item_name == "coffee_packs":
                        sleep_r.hours_awake = max(0, sleep_r.hours_awake - 5)
                    st.success(f"Used {label}")
                else:
                    st.error(f"Out of {label}")

        st.markdown('<div class="env-label">— GENERATOR —</div>', unsafe_allow_html=True)
        if st.button("⬡ TOGGLE GENERATOR"):
            state = power.toggle_generator()
            log.log(EventSeverity.INFO, "POWER",
                     f"Generator {'STARTED' if state else 'STOPPED'}",
                     int(st.session_state.day), ingame_h)
            st.info(f"Generator: {'RUNNING' if state else 'OFF'}")

    with col_main:
        if st.button("▶ ADVANCE TIME & UPDATE ALL SYSTEMS", use_container_width=True):
            rng_env = np.random.default_rng(int(time.time_ns()) % (2**31))

            # Advance weather
            new_weather = wx_sim.step(ingame_h)
            st.session_state.current_weather = new_weather
            weather = new_weather

            # Degrade dish
            dish.degrade(delta_h, weather, rng_env)

            # Update power
            solar_el = AstronomicalCalc.solar_elevation(
                STATION_LATITUDE_DEG, STATION_LONGITUDE_DEG, ingame_h, int(day_of_year))
            day_phase_enum = AstronomicalCalc.day_phase(solar_el)
            power_events = power.update(delta_h, weather, day_phase_enum)

            # Update sleep
            sleep_r.update(delta_h)

            # Day advance
            if ingame_h + delta_h >= 24:
                st.session_state.day = int(st.session_state.day) + 1
                inv.consume("food_rations", 1)

            # Compute signal quality
            quality = sq.compute(dish, weather, power, sleep_r,
                                  el_deg, freq_mhz, int(day_of_year), ingame_h)

            # Sample random events
            game_state = {
                "dish_health_fraction": dish.effective_area_fraction,
                "generator_fuel_l":     power.generator_fuel_l,
                "hours_awake":          sleep_r.hours_awake,
                "weather_state":        weather.state.value,
                "ingame_hour":          ingame_h,
            }
            new_events = events.sample(delta_h, game_state)

            # Apply event effects
            for ev in new_events:
                eff = ev.effects
                if "surface_contamination" in eff:
                    dish.surface_contamination = float(np.clip(
                        dish.surface_contamination + eff["surface_contamination"], 0, 1))
                if "lna_noise_figure_db" in eff:
                    dish.lna_noise_figure_db = float(np.clip(
                        dish.lna_noise_figure_db + eff["lna_noise_figure_db"], 0.3, 5))
                if "az_motor_health" in eff:
                    dish.az_motor_health = float(np.clip(
                        dish.az_motor_health + eff["az_motor_health"], 0, 1))
                if "connector_vswr" in eff:
                    dish.connector_vswr = float(np.clip(
                        dish.connector_vswr + eff["connector_vswr"], 1.0, 5.0))
                if "feed_alignment_deg" in eff:
                    dish.feed_alignment_deg = float(
                        dish.feed_alignment_deg + eff["feed_alignment_deg"])
                if "shrimp_packs" in eff:
                    inv.add("shrimp_packs", eff["shrimp_packs"])
                if "food_rations" in eff:
                    inv.add("food_rations", eff["food_rations"])
                if "blank_drives" in eff:
                    inv.add("blank_drives", eff["blank_drives"])
                if "cleaning_kits" in eff:
                    inv.add("cleaning_kits", eff["cleaning_kits"])
                if "hours_awake_penalty" in eff:
                    sleep_r.hours_awake += eff["hours_awake_penalty"]
                if "void_fragments" in eff:
                    inv.add("void_fragments", eff["void_fragments"])
                if "anomalous_notes" in eff:
                    inv.add("anomalous_notes", eff["anomalous_notes"])

                log.log(ev.severity, ev.category, f"{ev.title}: {ev.description}",
                         int(st.session_state.day), ingame_h)

            # Log power events
            for pev_key in power_events:
                log.log(EventSeverity.CRITICAL if "outage" in pev_key else EventSeverity.WARNING,
                         "POWER", pev_key.replace("_", " ").upper(),
                         int(st.session_state.day), ingame_h)

            # Dashboard plot
            fig_dash = viz.plot_station_dashboard(dish, power, sleep_r, weather, quality)
            st.pyplot(fig_dash, use_container_width=True)
            plt.close(fig_dash)

            # Key metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Dish Eff",   f"{dish.effective_area_fraction*100:.0f}%",
                       delta=dish.overall_status.value)
            m2.metric("Battery",    f"{power.battery_fraction*100:.0f}%")
            m3.metric("Fuel",       f"{power.generator_fuel_l:.0f} L")
            m4.metric("Fatigue",    sleep_r.fatigue_level.value)
            m5.metric("Composite Q", f"{quality['composite']:.3f}")
            m6.metric("SNR Penalty", f"{quality['snr_penalty_db']:.1f} dB")

            # Events display
            if new_events:
                st.markdown('<div class="env-label">— NEW EVENTS —</div>',
                            unsafe_allow_html=True)
                for ev in new_events:
                    css = {
                        EventSeverity.HORROR:   "event-horror",
                        EventSeverity.CRITICAL: "event-critical",
                        EventSeverity.WARNING:  "event-warning",
                        EventSeverity.INFO:     "event-info",
                    }.get(ev.severity, "event-info")
                    st.markdown(
                        f'<div class="{css}">'
                        f'[{ev.severity.name}] {ev.title}<br>'
                        f'<span style="color:#88ffcc">{ev.description}</span><br>'
                        f'<span style="color:#aaffcc">↳ {ev.resolution}</span>'
                        f'</div>',
                        unsafe_allow_html=True)

            # Critical shortages
            shortages = inv.critical_shortages()
            if shortages:
                st.markdown('<div class="env-label">— CRITICAL SHORTAGES —</div>',
                            unsafe_allow_html=True)
                for s in shortages:
                    st.markdown(f'<span class="shortage-badge">⚠ {s}</span>',
                                unsafe_allow_html=True)

        # ── Tabs ────────────────────────────────────────────────────────────
        tabs = st.tabs(["[ INVENTORY ]",
                        "[ WEATHER HISTORY ]",
                        "[ STATION LOG ]",
                        "[ SIGNAL QUALITY ]",
                        "[ ASTRONOMICAL ]"])

        with tabs[0]:
            inv_dict = inv.to_dict()
            inv_rows = []
            for k, v in inv_dict.items():
                status = ("🔴 CRITICAL" if isinstance(v, int) and v == 0 else
                           "⚠ LOW"      if isinstance(v, int) and v < 3  else
                           "✓ OK")
                inv_rows.append({
                    "Item":   k.replace("_", " ").title(),
                    "Count":  v,
                    "Status": status,
                })
            st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)

        with tabs[1]:
            weather_df = wx_sim.weather_history_df()
            if not weather_df.empty:
                fig_wx = viz.plot_weather_history(weather_df)
                st.pyplot(fig_wx, use_container_width=True)
                plt.close(fig_wx)
                st.dataframe(weather_df.tail(20), use_container_width=True, hide_index=True)

                # Forecast
                forecast = wx_sim.forecast_severity(n_steps=48)
                st.markdown('<div class="env-label">— 24H WEATHER FORECAST (MC) —</div>',
                            unsafe_allow_html=True)
                fc_df = pd.DataFrame(
                    sorted(forecast.items(), key=lambda x: -x[1]),
                    columns=["State", "Probability"]
                )
                fc_df["Probability %"] = (fc_df["Probability"] * 100).round(1)
                st.dataframe(fc_df[["State","Probability %"]],
                             use_container_width=True, hide_index=True)

        with tabs[2]:
            log_df = log.recent(30)
            if not log_df.empty:
                st.dataframe(log_df, use_container_width=True, hide_index=True)
                st.markdown('<div class="env-label">— EVENT HISTORY —</div>',
                            unsafe_allow_html=True)
                ev_df = events.event_history_df()
                if not ev_df.empty:
                    st.dataframe(ev_df, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="env-label">NO LOG ENTRIES YET.</div>',
                            unsafe_allow_html=True)

        with tabs[3]:
            quality_now = sq.compute(dish, weather, power, sleep_r,
                                      el_deg, freq_mhz, int(day_of_year), ingame_h)
            q_df = pd.DataFrame([
                {"Factor": k.replace("_", " ").title(),
                 "Value":  round(v, 4),
                 "Status": ("✓" if v > 0.75 else "⚠" if v > 0.5 else "✗")}
                for k, v in quality_now.items()
            ])
            st.dataframe(q_df, use_container_width=True, hide_index=True)

            # Elevation sweep
            el_range = np.linspace(5, 90, 50)
            comp_vs_el = []
            for el in el_range:
                qe = sq.compute(dish, weather, power, sleep_r,
                                 el, freq_mhz, int(day_of_year), ingame_h)
                comp_vs_el.append(qe["composite"])
            fig_el, ax_el = plt.subplots(figsize=(8, 2.5), facecolor=viz.PAL["bg"])
            ax_el.plot(el_range, comp_vs_el, color=viz.PAL["fg"], lw=1.0)
            ax_el.fill_between(el_range, comp_vs_el, 0, alpha=0.1, color=viz.PAL["fg"])
            ax_el.axvline(el_deg, color=viz.PAL["acc"], lw=0.8, ls="--",
                           label=f"Current el={el_deg:.0f}°")
            ax_el.set_xlabel("Elevation (°)", fontsize=6, color=viz.PAL["fg"])
            ax_el.set_ylabel("Composite Q", fontsize=6, color=viz.PAL["fg"])
            ax_el.set_facecolor(viz.PAL["bg"])
            ax_el.tick_params(colors=viz.PAL["axis"], labelsize=6)
            for sp in ax_el.spines.values(): sp.set_edgecolor(viz.PAL["dim"])
            ax_el.set_title("QUALITY VS ELEVATION", color=viz.PAL["fg"],
                             fontsize=7.5, loc="left", fontfamily="monospace")
            ax_el.legend(fontsize=6, facecolor=viz.PAL["bg"],
                          edgecolor=viz.PAL["dim"], labelcolor=viz.PAL["fg"])
            st.pyplot(fig_el, use_container_width=True)
            plt.close(fig_el)

        with tabs[4]:
            solar_el = AstronomicalCalc.solar_elevation(
                STATION_LATITUDE_DEG, STATION_LONGITUDE_DEG,
                ingame_h, int(day_of_year))
            day_ph = AstronomicalCalc.day_phase(solar_el)
            t_gal  = AstronomicalCalc.galactic_noise_k(el_deg, freq_mhz)
            iono   = AstronomicalCalc.ionospheric_scintillation_factor(
                el_deg, ingame_h, int(day_of_year))

            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Solar Elevation", f"{solar_el:.1f}°")
            ac2.metric("Day Phase",       day_ph.value)
            ac3.metric("Galactic T",      f"{t_gal:.1f} K")
            ac4.metric("Iono Scint",      f"{iono:.3f}")

            # 24h solar elevation plot
            hours_24 = np.linspace(0, 24, 200)
            elevations = [AstronomicalCalc.solar_elevation(
                STATION_LATITUDE_DEG, STATION_LONGITUDE_DEG, h, int(day_of_year))
                for h in hours_24]
            fig_sun, ax_sun = plt.subplots(figsize=(9, 2.5), facecolor=viz.PAL["bg"])
            el_arr = np.array(elevations)
            above = np.where(el_arr > 0, el_arr, 0)
            below = np.where(el_arr <= 0, el_arr, 0)
            ax_sun.plot(hours_24, el_arr, color=viz.PAL["gold"], lw=1.0)
            ax_sun.fill_between(hours_24, above, 0, alpha=0.2, color=viz.PAL["gold"])
            ax_sun.fill_between(hours_24, below, 0, alpha=0.15, color=viz.PAL["purp"])
            ax_sun.axhline(0,   color=viz.PAL["dim"], lw=0.8, ls="--")
            ax_sun.axhline(-18, color=viz.PAL["purp"], lw=0.6, ls=":", alpha=0.7,
                            label="Astro night (-18°)")
            ax_sun.axvline(ingame_h, color=viz.PAL["acc"], lw=0.8, ls="--",
                            label=f"Now ({ingame_h:.1f}h)")
            ax_sun.set_xlabel("Hour UTC", fontsize=6, color=viz.PAL["fg"])
            ax_sun.set_ylabel("Solar Elevation (°)", fontsize=6, color=viz.PAL["fg"])
            ax_sun.set_facecolor(viz.PAL["bg"])
            ax_sun.tick_params(colors=viz.PAL["axis"], labelsize=6)
            for sp in ax_sun.spines.values(): sp.set_edgecolor(viz.PAL["dim"])
            ax_sun.set_title(f"SOLAR ELEVATION — DAY {int(day_of_year)} — LAT {STATION_LATITUDE_DEG}°N",
                              color=viz.PAL["fg"], fontsize=7.5, loc="left",
                              fontfamily="monospace")
            ax_sun.legend(fontsize=6, facecolor=viz.PAL["bg"],
                           edgecolor=viz.PAL["dim"], labelcolor=viz.PAL["fg"])
            ax_sun.grid(True, color=viz.PAL["grid"], alpha=0.4, ls=":")
            st.pyplot(fig_sun, use_container_width=True)
            plt.close(fig_sun)

            # Galactic noise vs frequency at current elevation
            freqs_range = np.logspace(2, 4.4, 100)
            t_gal_range = [AstronomicalCalc.galactic_noise_k(el_deg, f) for f in freqs_range]
            fig_gal, ax_gal = plt.subplots(figsize=(9, 2.5), facecolor=viz.PAL["bg"])
            ax_gal.loglog(freqs_range, t_gal_range, color=viz.PAL["fg"], lw=1.0)
            ax_gal.axvline(freq_mhz, color=viz.PAL["acc"], lw=0.8, ls="--",
                            label=f"{freq_mhz:.1f} MHz")
            ax_gal.axvline(1420.4, color=viz.PAL["gold"], lw=0.7, ls=":", alpha=0.7,
                            label="H-I 1420 MHz")
            ax_gal.set_xlabel("Frequency (MHz)", fontsize=6, color=viz.PAL["fg"])
            ax_gal.set_ylabel("T_gal (K)", fontsize=6, color=viz.PAL["fg"])
            ax_gal.set_facecolor(viz.PAL["bg"])
            ax_gal.tick_params(colors=viz.PAL["axis"], labelsize=6)
            for sp in ax_gal.spines.values(): sp.set_edgecolor(viz.PAL["dim"])
            ax_gal.set_title("GALACTIC NOISE TEMPERATURE vs FREQUENCY",
                              color=viz.PAL["fg"], fontsize=7.5, loc="left",
                              fontfamily="monospace")
            ax_gal.legend(fontsize=6, facecolor=viz.PAL["bg"],
                           edgecolor=viz.PAL["dim"], labelcolor=viz.PAL["fg"])
            ax_gal.grid(True, color=viz.PAL["grid"], alpha=0.4, ls=":")
            st.pyplot(fig_gal, use_container_width=True)
            plt.close(fig_gal)


if __name__ == "__main__":
    environment_monitor_page()
