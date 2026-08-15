"""
Regression tests locking the dataset facts and the prototype refusals.

These are the claims the reports rest on. If SILSO changes a file's structure, or
a dependency changes a behaviour, these should be what fails first.

Run:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silso_uncertainty import reader  # noqa: E402
from silso_uncertainty.uncertain import (FULLY_CORRELATED, INDEPENDENT,  # noqa: E402
                                         CorrelationModel, UncertainTimeSeries)


@pytest.fixture(scope="module")
def daily():
    frame, _ = reader.read_daily_total()
    return frame


@pytest.fixture(scope="module")
def monthly():
    frame, _ = reader.read_monthly_total()
    return frame


# -- dataset facts --------------------------------------------------------
def test_zero_is_not_missing(daily):
    """Genuine zero sunspot numbers survive parsing and are not masked."""
    assert (daily["sunspot_number"] == 0).sum() > 10000
    assert not daily.loc[daily["sunspot_number"] == 0, "sunspot_number"].isna().any()


def test_sentinel_masked_to_nan(daily):
    assert (daily["sunspot_number"] == -1).sum() == 0
    assert (daily["sigma"] == -1).sum() == 0
    assert daily["sunspot_number"].isna().sum() > 0


def test_monthly_sigma_missing_while_value_valid(monthly):
    """The case joint masking would get wrong: valid mean, missing sigma."""
    both = monthly["sunspot_number"].notna() & monthly["sigma"].isna()
    assert both.sum() > 800
    # ...and all of them precede the start of the daily record.
    assert monthly.index[both].max().year < 1818


def test_hemispheric_counts_masked_independently():
    frame, _ = reader.read_daily_hemispheric()
    assert frame["sunspot_number"].notna().all()
    assert frame["sigma"].notna().all()
    missing = frame["n_observations_north"].isna()
    assert missing.sum() > 8000
    assert frame.index[missing].max().year < 2015


def test_smoothed_edges_only():
    frame, _ = reader.read_monthly_smoothed()
    missing = frame["sunspot_number_smoothed"].isna()
    assert missing.sum() == 12, "13-month filter loses exactly 6 months at each end"


def test_published_monthly_sigma_is_pooled_dispersion(daily, monthly):
    """The keystone result: SILSO's documented pooled formula reproduces sigma."""
    valid = daily.dropna(subset=["sunspot_number", "sigma", "n_observations"])
    num = (valid["n_observations"] * valid["sigma"] ** 2).resample("MS").sum()
    den = valid["n_observations"].resample("MS").sum()
    pooled = np.sqrt(num / den)

    both = pooled.to_frame("pooled").join(monthly["sigma"].rename("published"),
                                          how="inner").dropna()
    assert len(both) > 2400
    assert (both["pooled"] - both["published"]).abs().max() <= 0.05


def test_monthly_counts_are_sums_of_daily(daily, monthly):
    valid = daily.dropna(subset=["n_observations"])
    summed = valid["n_observations"].resample("MS").sum()
    both = summed.to_frame("summed").join(
        monthly["n_observations"].rename("published"), how="inner").dropna()
    assert (both["summed"] == both["published"]).all()


def test_single_station_era_ends_in_1981(daily):
    multi = daily[daily["n_observations"] > 1]
    assert multi.index.min() == np.datetime64("1981-01-01")


# -- prototype behaviour ----------------------------------------------------
def test_resample_refuses_without_correlation():
    u = UncertainTimeSeries.from_silso("daily_total")
    with pytest.raises(ValueError, match="explicit `correlation`"):
        u.resample("MS")


def test_resample_refuses_raw_dispersion():
    u = UncertainTimeSeries.from_silso("daily_total")
    with pytest.raises(ValueError, match="station dispersion"):
        u.resample("MS", correlation=INDEPENDENT)


def test_correlation_endpoints_differ_by_sqrt_n():
    """rho=0 and rho=1 differ by ~sqrt(N), which is the whole design argument."""
    s = np.full(30, 2.0)
    indep = INDEPENDENT.combine_mean(s)
    full = FULLY_CORRELATED.combine_mean(s)
    assert indep == pytest.approx(2.0 / np.sqrt(30))
    assert full == pytest.approx(2.0)
    assert full / indep == pytest.approx(np.sqrt(30))


def test_correlation_bounds_checked():
    with pytest.raises(ValueError):
        CorrelationModel(1.5)


def test_as_standard_error_divides_by_sqrt_n():
    u = UncertainTimeSeries.from_silso("daily_total")
    sem = u.as_standard_error()
    frame = sem.to_dataframe()
    ratio = (u.sigma() / sem.sigma()).dropna()
    expected = np.sqrt(frame["n_observations"]).dropna()
    common = ratio.index.intersection(expected.index)
    assert np.allclose(ratio.loc[common], expected.loc[common])
    assert sem.estimand["sunspot_number"] == "standard_error"
