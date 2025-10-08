import io
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from speed_profiles import (
    load_speed_profiles,
    filter_speed_range,
    compute_speed_statistics,
    compute_speed_percentiles,
    rolling_average_speed,
)


def test_load_speed_profiles_with_converters_and_date_parsing():
    csv_content = """timestamp,speed,lane\n2024-01-01T08:00:00,45.5,1\n2024-01-01T08:05:00,50.2,2\n"""
    buffer = io.StringIO(csv_content)
    records = load_speed_profiles(buffer, converters={"lane": int})
    assert len(records) == 2
    assert all(isinstance(row["timestamp"], datetime) for row in records)
    assert records[0]["lane"] == 1
    assert records[1]["speed"] == 50.2


def test_filter_speed_range_filters_invalid_rows():
    records = [
        {"speed": 45},
        {"speed": "50"},
        {"speed": None},
        {"speed": 65},
        {"speed": 75},
    ]
    filtered = filter_speed_range(records, min_speed=50, max_speed=70)
    assert filtered == [{"speed": 50.0}, {"speed": 65.0}]


def test_compute_speed_statistics_and_percentiles():
    records = [{"speed": value} for value in [55, 60, 65, 70, 75]]
    stats = compute_speed_statistics(records)
    assert stats["count"] == pytest.approx(5)
    assert stats["mean"] == pytest.approx(65)
    assert stats["median"] == pytest.approx(65)
    assert stats["min"] == pytest.approx(55)
    assert stats["max"] == pytest.approx(75)
    assert stats["stdev"] == pytest.approx(7.90569, rel=1e-5)

    percentiles = compute_speed_percentiles(records, [0, 25, 50, 75, 100])
    assert percentiles[0] == pytest.approx(55)
    assert percentiles[25] == pytest.approx(60)
    assert percentiles[50] == pytest.approx(65)
    assert percentiles[75] == pytest.approx(70)
    assert percentiles[100] == pytest.approx(75)


def test_rolling_average_speed_with_variable_window():
    records = [{"speed": value} for value in [40, 45, 50, 55]]
    rolling = rolling_average_speed(records, window=2)
    assert rolling == [40.0, 42.5, 47.5, 52.5]

    with pytest.raises(ValueError):
        rolling_average_speed(records, window=0)
