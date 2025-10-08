"""Utilities for working with roadway speed profile data."""

from .analysis import (
    load_speed_profiles,
    filter_speed_range,
    compute_speed_statistics,
    rolling_average_speed,
    compute_speed_percentiles,
)

__all__ = [
    "load_speed_profiles",
    "filter_speed_range",
    "compute_speed_statistics",
    "rolling_average_speed",
    "compute_speed_percentiles",
]
