"""Rayleigh-wave back-azimuth, filtering, and orbit-detection tools."""

from .rayleigh_backazimuth import estimate_rayleigh_baz_posterior
from .rayleigh_filter import (
    add_particleman_nip_energy_masks,
    simple_elliptic_filter_rayleigh,
    simple_stockwell_filter_rayleigh,
)
from .rayleigh_orbit_detector import detect_rayleigh_orbits_from_stockwell
from .prem_dispersion import RayleighDispersion, rayleigh_group_velocity_prem

__version__ = "0.1.1"

__all__ = [
    "estimate_rayleigh_baz_posterior",
    "simple_elliptic_filter_rayleigh",
    "simple_stockwell_filter_rayleigh",
    "add_particleman_nip_energy_masks",
    "detect_rayleigh_orbits_from_stockwell",
    "RayleighDispersion",
    "rayleigh_group_velocity_prem",
    "__version__",
]
