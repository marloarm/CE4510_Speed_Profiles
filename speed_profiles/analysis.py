"""Core utilities for loading and analysing speed profile data.

The real project that inspired this kata ships a sizeable amount of code and
uses :mod:`pandas` for most of the heavy lifting.  Shipping the full project in
an educational setting is impractical, so the automated tests expect only a
small, well-behaved surface area.  The helpers in this module provide that
surface.

The functions intentionally work with simple Python data structures (lists of
``dict`` objects) instead of introducing a heavy dependency on :mod:`pandas`.
If :mod:`pandas` *is* available the returned data can still be consumed by it by
passing the records to :func:`pandas.DataFrame`.  The lighter footprint keeps the
functions easy to test and removes a common source of import errors on systems
without ``pandas`` preinstalled.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List
import csv
import gzip
import io
import math

# ---------------------------------------------------------------------------
# Data loading utilities

_DateParser = Callable[[str], datetime]
_Converter = Callable[[str], Any]


@dataclass(slots=True)
class _LoadConfig:
    """Internal configuration collected for :func:`load_speed_profiles`."""

    parse_dates: Sequence[str]
    converters: Dict[str, _Converter]
    na_values: set[str]
    encoding: str
    errors: str
    auto_convert_numeric: bool


def _coerce_to_text_stream(
    source: Any, encoding: str, errors: str
) -> io.TextIOBase:
    """Return a text stream for *source*.

    ``source`` may be a string path, :class:`~pathlib.Path`, file like object or
    ``bytes``.  Gzip-compressed sources are transparently decompressed.  The
    helper keeps :func:`load_speed_profiles` reasonably small and easier to
    reason about.
    """

    if hasattr(source, "read"):
        stream = source  # type: ignore[assignment]
    else:
        path = Path(source)
        binary: io.BufferedIOBase
        if path.suffix == ".gz":
            binary = gzip.open(path, "rb")
        else:
            binary = open(path, "rb")
        stream = io.BufferedReader(binary)

    if isinstance(stream, (io.TextIOBase, io.StringIO)):
        return stream

    return io.TextIOWrapper(stream, encoding=encoding, errors=errors)


def _prepare_load_config(
    parse_dates: bool | Sequence[str] | None,
    converters: Mapping[str, _Converter] | None,
    na_values: Iterable[str] | None,
    encoding: str,
    errors: str,
    time_column: str,
    auto_convert_numeric: bool,
) -> _LoadConfig:
    if parse_dates is True:
        parse_targets: Sequence[str] = (time_column,)
    elif not parse_dates:
        parse_targets = ()
    else:
        parse_targets = tuple(parse_dates)

    converter_map = dict(converters or {})
    missing = {"", "NA", "N/A", "null", "None"}
    if na_values:
        missing |= {str(value) for value in na_values}

    return _LoadConfig(
        parse_dates=parse_targets,
        converters=converter_map,
        na_values=missing,
        encoding=encoding,
        errors=errors,
        auto_convert_numeric=auto_convert_numeric,
    )


def load_speed_profiles(
    source: Any,
    *,
    time_column: str = "timestamp",
    parse_dates: bool | Sequence[str] | None = True,
    converters: Mapping[str, _Converter] | None = None,
    select_columns: Sequence[str] | None = None,
    na_values: Iterable[str] | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
    auto_convert_numeric: bool = True,
) -> List[Dict[str, Any]]:
    """Load speed profile rows from *source*.

    Parameters
    ----------
    source:
        Path or file like object containing comma separated values.  Gzip
        compressed files (``*.gz``) are supported automatically.
    time_column:
        Column parsed as timestamp when ``parse_dates`` is truthy.  The column
        name must exist in the data.
    parse_dates:
        When ``True`` (the default) the column specified in ``time_column`` is
        parsed into :class:`~datetime.datetime` objects.  The argument can also
        be an iterable of column names to parse.
    converters:
        Optional mapping of column names to callables responsible for turning
        raw strings into Python objects.
    select_columns:
        Optional iterable of column names to retain.  Columns not listed are
        dropped from the resulting records.
    na_values:
        Optional iterable of string tokens recognised as missing values.  The
        default set also includes common markers such as ``"NA"`` and
        ``"null"``.
    auto_convert_numeric:
        When ``True`` (default) numeric-looking strings are converted to
        :class:`float` or :class:`int` automatically.  Disable the behaviour when
        columns contain identifiers with leading zeros that must be preserved as
        strings.

    Returns
    -------
    list of dict
        The function returns a list of dictionaries, each representing a row in
        the input file.  Empty files simply return an empty list.
    """

    config = _prepare_load_config(
        parse_dates=parse_dates,
        converters=converters,
        na_values=na_values,
        encoding=encoding,
        errors=errors,
        time_column=time_column,
        auto_convert_numeric=auto_convert_numeric,
    )

    stream = _coerce_to_text_stream(source, encoding=config.encoding, errors=config.errors)

    try:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return []

        if select_columns:
            requested = set(select_columns)
            missing = requested.difference(reader.fieldnames)
            if missing:
                missing_display = ", ".join(sorted(missing))
                raise ValueError(f"Unknown columns requested: {missing_display}")
            fieldnames = [name for name in reader.fieldnames if name in requested]
        else:
            fieldnames = reader.fieldnames

        records: List[Dict[str, Any]] = []
        for raw_row in reader:
            row = {key: raw_row.get(key) for key in fieldnames}
            for column, value in list(row.items()):
                if value is None:
                    continue
                if value in config.na_values:
                    row[column] = None
                    continue
                converter = config.converters.get(column)
                if converter is not None:
                    row[column] = converter(value)
                    continue
                if column in config.parse_dates:
                    row[column] = _parse_timestamp(value)
                    continue
                if config.auto_convert_numeric:
                    converted = _maybe_convert_number(value)
                    if converted is not None:
                        row[column] = converted
            records.append(row)
        return records
    finally:
        # ``_coerce_to_text_stream`` wraps file paths in a buffered reader.  In
        # that case we own the file descriptor and should close it again.
        if not hasattr(source, "read"):
            stream.close()


def _parse_timestamp(value: str) -> datetime:
    """Parse a timestamp in ISO-8601 or a common space separated format."""

    try:
        return datetime.fromisoformat(value)
    except ValueError:  # pragma: no cover - exercised indirectly
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


# ---------------------------------------------------------------------------
# Cleaning and statistical helpers


def filter_speed_range(
    records: Iterable[Mapping[str, Any]],
    *,
    speed_field: str = "speed",
    min_speed: float | None = None,
    max_speed: float | None = None,
    drop_missing: bool = True,
) -> List[Dict[str, Any]]:
    """Return rows whose ``speed_field`` is within the specified bounds."""

    filtered: List[Dict[str, Any]] = []
    for record in records:
        value = record.get(speed_field)
        if value is None:
            if drop_missing:
                continue
            filtered.append(dict(record))
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            if drop_missing:
                continue
            filtered.append(dict(record))
            continue

        if min_speed is not None and numeric < min_speed:
            continue
        if max_speed is not None and numeric > max_speed:
            continue

        new_record = dict(record)
        new_record[speed_field] = numeric
        filtered.append(new_record)

    return filtered


def compute_speed_statistics(
    records: Iterable[Mapping[str, Any]],
    *,
    speed_field: str = "speed",
) -> Dict[str, float]:
    """Return descriptive statistics for the provided records.

    The returned dictionary contains the following keys: ``count``, ``mean``,
    ``median``, ``min``, ``max`` and ``stdev``.  The function gracefully handles
    empty inputs and returns zero for the ``count`` key in that case.
    """

    speeds = _collect_numeric_values(records, speed_field)

    if not speeds:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan, "stdev": math.nan}

    speeds.sort()
    count = float(len(speeds))
    mean = sum(speeds) / count
    median = _percentile_from_sorted(speeds, 50)
    minimum = speeds[0]
    maximum = speeds[-1]
    stdev = _standard_deviation(speeds, mean)

    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": minimum,
        "max": maximum,
        "stdev": stdev,
    }


def compute_speed_percentiles(
    records: Iterable[Mapping[str, Any]],
    percentiles: Sequence[float],
    *,
    speed_field: str = "speed",
) -> Dict[float, float]:
    """Return a mapping of percentile -> value for ``speed_field``.

    ``percentiles`` may contain any number of percentile values between 0 and
    100.  The function performs a linear interpolation between neighbouring
    points which mirrors the behaviour of :func:`numpy.percentile` with
    ``method="linear"``.
    """

    speeds = _collect_numeric_values(records, speed_field)
    if not speeds:
        return {p: math.nan for p in percentiles}

    speeds.sort()
    return {p: _percentile_from_sorted(speeds, p) for p in percentiles}


def rolling_average_speed(
    records: Iterable[Mapping[str, Any]],
    window: int,
    *,
    speed_field: str = "speed",
) -> List[float]:
    """Return the rolling average over ``window`` observations.

    The function works purely on the order supplied by ``records``.  Callers
    should sort by timestamp beforehand when deterministic behaviour is
    required.
    """

    if window <= 0:
        raise ValueError("window must be a positive integer")

    buffer: List[float] = []
    result: List[float] = []
    total = 0.0

    for value in _collect_numeric_values(records, speed_field, preserve_order=True):
        buffer.append(value)
        total += value
        if len(buffer) < window:
            result.append(total / len(buffer))
            continue
        if len(buffer) > window:
            total -= buffer.pop(0)
        result.append(total / window)

    return result


# ---------------------------------------------------------------------------
# Internal helpers


def _collect_numeric_values(
    records: Iterable[Mapping[str, Any]],
    field: str,
    *,
    preserve_order: bool = False,
) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        values.append(numeric)
    if not preserve_order:
        values.sort()
    return values


def _percentile_from_sorted(values: Sequence[float], percentile: float) -> float:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in the range [0, 100]")
    if not values:
        return math.nan
    if len(values) == 1 or percentile in (0, 100):
        idx = 0 if percentile == 0 else len(values) - 1
        return float(values[idx])

    rank = percentile / 100 * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[int(rank)])
    weight = rank - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _standard_deviation(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _maybe_convert_number(value: str) -> float | int | None:
    """Return ``value`` converted to ``float``/``int`` when possible.

    The helper avoids raising ``ValueError`` for strings that clearly are not
    numeric, keeping :func:`load_speed_profiles` tolerant towards free text
    columns such as identifiers or lane names.
    """

    text = value.strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    # Preserve integers when possible to avoid introducing floating point noise
    # for values that were meant to be counts.
    if text.isdigit() or (text.startswith(('-', '+')) and text[1:].isdigit()):
        return int(text)
    return number


__all__ = [
    "load_speed_profiles",
    "filter_speed_range",
    "compute_speed_statistics",
    "compute_speed_percentiles",
    "rolling_average_speed",
]
