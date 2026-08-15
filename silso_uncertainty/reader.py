"""
Parsers for the SILSO version-2 sunspot-number files.

Every structural fact encoded here was verified against the actual bytes of the
files (see section 1 of the README), not taken from the info pages.

Sentinel handling
-----------------
SILSO uses ``-1`` as "no value available". This is semantically distinct from a
real zero, which is extremely common in this dataset: 11,398 days in the daily
total file have a genuine sunspot number of exactly 0. Any reader that coerces
-1 to 0, or that treats 0 as missing, is wrong in both directions.

The sentinel is handled *independently per column*, because the columns do not
fail together in every file:

* ``SN_d_tot``: value and sigma are -1 on exactly the same 3,247 rows.
* ``SN_m_tot``: 828 rows have a valid monthly mean but sigma == -1 (every
  month before 1818-01, i.e. before per-station daily data exists).
* ``SN_d_hem``: no missing values or sigmas at all; -1 appears only in the two
  hemispheric station-count columns, on every day before 2015-01-01.

so a reader that assumes the columns are missing together silently invents data
for 828 months. Each column is masked on its own.
"""
from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd

from .fetch import fetch

#: SILSO's missing-data sentinel.
SENTINEL = -1

#: The sunspot number is a dimensionless index; station counts are counts.
SSN = u.dimensionless_unscaled
COUNT = u.count


def _mask_sentinel(series: pd.Series) -> pd.Series:
    """Replace SILSO's -1 sentinel with NaN, leaving genuine zeros alone."""
    out = series.astype("float64")
    return out.mask(out == SENTINEL)


def _daily_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Build a DatetimeIndex from the explicit Y/M/D columns.

    The fractional-year column (col 4) is *not* used to build the index. It is a
    derived convenience column carrying only 3 decimal places, which is roughly
    0.4 day of resolution, not enough to round-trip a daily date unambiguously.
    The integer Y/M/D columns are exact, so they are authoritative here and the
    fractional year is retained only as a passthrough column.
    """
    return pd.DatetimeIndex(
        pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=frame["day"])),
        name="time",
    )


def read_daily_total(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Read ``SN_d_tot_V2.0.csv`` (daily total sunspot number, 1818-01-01 onward).

    Returns
    -------
    frame : pandas.DataFrame
        Indexed by date, with columns ``sunspot_number``, ``sigma``,
        ``n_observations``, ``definitive``, ``year_fraction``.
    units : dict
        Mapping of column name to `astropy.units.Unit`, in the shape
        `sunpy.timeseries.GenericTimeSeries` expects.
    """
    path = path or fetch("daily_total")
    frame = pd.read_csv(
        path, sep=";", header=None,
        names=["year", "month", "day", "year_fraction",
               "sunspot_number", "sigma", "n_observations", "definitive"],
    )
    index = _daily_index(frame)

    out = pd.DataFrame(index=index)
    out["sunspot_number"] = _mask_sentinel(frame["sunspot_number"]).to_numpy()
    out["sigma"] = _mask_sentinel(frame["sigma"]).to_numpy()
    # n_observations uses 0 (not -1) to mark "no observation"; 0 stations is
    # genuinely the same statement as "missing", so it is masked to NaN too.
    n_obs = frame["n_observations"].astype("float64")
    out["n_observations"] = n_obs.mask(n_obs == 0).to_numpy()
    out["definitive"] = frame["definitive"].astype(bool).to_numpy()
    out["year_fraction"] = frame["year_fraction"].to_numpy()

    units = {
        "sunspot_number": SSN,
        "sigma": SSN,
        "n_observations": COUNT,
        "year_fraction": u.year,
    }
    return out, units


def read_daily_hemispheric(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Read ``SN_d_hem_V2.0.csv`` (daily hemispheric sunspot numbers, 1992 onward).

    The two hemispheric station-count columns are -1 for every day before
    2015-01-01, on rows whose values and sigmas are perfectly valid. This is the
    clearest case in the dataset of a per-column sentinel.
    """
    path = path or fetch("daily_hemispheric")
    frame = pd.read_csv(
        path, sep=";", header=None,
        names=["year", "month", "day", "year_fraction",
               "sunspot_number", "sunspot_number_north", "sunspot_number_south",
               "sigma", "sigma_north", "sigma_south",
               "n_observations", "n_observations_north", "n_observations_south",
               "definitive"],
    )
    index = _daily_index(frame)

    out = pd.DataFrame(index=index)
    value_cols = ["sunspot_number", "sunspot_number_north", "sunspot_number_south",
                  "sigma", "sigma_north", "sigma_south",
                  "n_observations", "n_observations_north", "n_observations_south"]
    for col in value_cols:
        out[col] = _mask_sentinel(frame[col]).to_numpy()
    out["definitive"] = frame["definitive"].astype(bool).to_numpy()
    out["year_fraction"] = frame["year_fraction"].to_numpy()

    units = {c: SSN for c in value_cols if not c.startswith("n_observations")}
    units.update({c: COUNT for c in value_cols if c.startswith("n_observations")})
    units["year_fraction"] = u.year
    return out, units


def _read_monthly(path, value_name: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(
        path, sep=";", header=None,
        names=["year", "month", "year_fraction", value_name, "sigma",
               "n_observations", "definitive"],
    )
    # Monthly rows are stamped at the *first* of the month. SILSO's own
    # fractional year points at mid-month, but a DatetimeIndex needs a single
    # convention and month-start is what pandas' "MS" resampling produces, so
    # the two are directly comparable.
    index = pd.DatetimeIndex(
        pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1)), name="time"
    )
    out = pd.DataFrame(index=index)
    out[value_name] = _mask_sentinel(frame[value_name]).to_numpy()
    out["sigma"] = _mask_sentinel(frame["sigma"]).to_numpy()
    out["n_observations"] = _mask_sentinel(frame["n_observations"]).to_numpy()
    out["definitive"] = frame["definitive"].astype(bool).to_numpy()
    out["year_fraction"] = frame["year_fraction"].to_numpy()

    units = {value_name: SSN, "sigma": SSN, "n_observations": COUNT,
             "year_fraction": u.year}
    return out, units


def read_monthly_total(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Read ``SN_m_tot_V2.0.csv`` (monthly mean total sunspot number, 1749 onward)."""
    return _read_monthly(path or fetch("monthly_total"), "sunspot_number")


def read_monthly_smoothed(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Read ``SN_ms_tot_V2.0.csv`` (13-month smoothed total sunspot number).

    The smoothed value is -1 on exactly 12 rows: the first six months of 1749 and
    the last six available months, because the 13-month filter needs +/- 6 months
    of context. The sigma column is -1 on those 12 rows *plus* the same 828
    pre-1818 rows as the monthly file.
    """
    return _read_monthly(path or fetch("monthly_smoothed"), "sunspot_number_smoothed")


def read_yearly_total(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Read ``SN_y_tot_V2.0.csv`` (yearly mean total sunspot number, 1700 onward)."""
    path = path or fetch("yearly_total")
    frame = pd.read_csv(
        path, sep=";", header=None,
        names=["year_fraction", "sunspot_number", "sigma", "n_observations", "definitive"],
    )
    # The yearly file has *no* Y/M/D columns, only a mid-year fractional year
    # (e.g. 1700.5), so here the fractional year is the only date information.
    year = np.floor(frame["year_fraction"]).astype(int)
    index = pd.DatetimeIndex(pd.to_datetime(dict(year=year, month=1, day=1)), name="time")

    out = pd.DataFrame(index=index)
    out["sunspot_number"] = _mask_sentinel(frame["sunspot_number"]).to_numpy()
    out["sigma"] = _mask_sentinel(frame["sigma"]).to_numpy()
    out["n_observations"] = _mask_sentinel(frame["n_observations"]).to_numpy()
    out["definitive"] = frame["definitive"].astype(bool).to_numpy()
    out["year_fraction"] = frame["year_fraction"].to_numpy()

    units = {"sunspot_number": SSN, "sigma": SSN, "n_observations": COUNT,
             "year_fraction": u.year}
    return out, units
