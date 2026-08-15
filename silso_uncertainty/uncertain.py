"""
A time series that treats uncertainty as first-class.

Design, and why
---------------
Three places uncertainty could live, and what each costs:

1. **An ordinary column** (see ``baseline.py``). Zero API change, and wrong in a
   specific way: nothing distinguishes ``sigma`` from a second measurement. Every
   operation that reduces rows (resample, mean, rebin) will happily take the
   arithmetic mean of the sigma column, which is not how uncertainty combines
   under any correlation assumption. It fails silently and plausibly.

2. **A parallel object** (a second series carrying sigma). Keeps the two in step
   only by convention. Any operation applied to one must be applied to the other
   by hand, and truncate/concatenate can desynchronise them without error.

3. **A typed attribute keyed by column**, which is what this module does and what
   `astropy.nddata` settled on. The uncertainty is stored as an
   `~astropy.nddata.NDUncertainty` instance, so it carries its own *type*
   (standard deviation vs variance vs inverse variance), and it is attached to
   the series rather than living in the data table, so row-reducing operations
   cannot mistake it for data.

The decisive argument for (3) is that `~astropy.nddata.NDUncertainty.propagate`
takes ``correlation`` as a **required positional argument**, and astropy now ships
a `Covariance` uncertainty class. Astropy concluded that propagation is not
well-defined without a stated correlation. The quadrature experiment in
``scripts/quadrature_check.py`` shows the same conclusion is forced by the SILSO
data itself.

This module adds one thing astropy's model does not have, which the SILSO data
turns out to require: an explicit record of the uncertainty's **estimand**.
SILSO's daily sigma is the dispersion of individual station counts, *not* the
standard error of the daily mean, and SILSO's published monthly sigma is a
pooled dispersion, again not a standard error. A reader that does not record
which of these it is holding cannot resample correctly, because the two
quantities scale differently with N. See section 3 of the README.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from astropy.nddata import StdDevUncertainty
from sunpy.timeseries import GenericTimeSeries

from . import reader
from .baseline import _numeric_only, plain_meta

#: What a sigma value actually estimates. SILSO ships both kinds in one dataset.
ESTIMANDS = {
    # Spread of the individual station reports contributing to one value.
    # Does NOT shrink as more values are averaged together.
    "station_dispersion",
    # Uncertainty on the mean itself. Shrinks as ~1/sqrt(N) under independence.
    "standard_error",
}


class CorrelationModel:
    """
    How the errors of the values being combined relate to each other.

    Combining ``N`` values with standard errors ``s_i`` into a mean gives

    .. math::
        \\mathrm{Var}(\\bar{x}) = \\frac{1}{N^2}
            \\left[ (1-\\rho)\\sum_i s_i^2 + \\rho \\left(\\sum_i s_i\\right)^2 \\right]

    under equicorrelation with coefficient ``rho``. The two limits bracket the
    design space and are both defensible defaults, which is exactly the problem:

    * ``rho = 0``: independent. Uncertainty averages down as 1/sqrt(N).
    * ``rho = 1``: fully correlated. Uncertainty does not average down at all.

    There is no neutral choice, so this prototype refuses to pick one implicitly.
    """

    def __init__(self, rho: float | Callable[[int], float] = 0.0, name: str | None = None):
        if not callable(rho) and not (-1.0 <= float(rho) <= 1.0):
            raise ValueError(f"correlation must lie in [-1, 1], got {rho}")
        self.rho = rho
        self.name = name or (f"callable:{rho.__name__}" if callable(rho) else f"rho={rho}")

    def rho_for(self, n: int) -> float:
        return float(self.rho(n)) if callable(self.rho) else float(self.rho)

    def combine_mean(self, standard_errors: np.ndarray) -> float:
        """Standard error of the mean of values with the given standard errors."""
        s = np.asarray(standard_errors, dtype="float64")
        s = s[np.isfinite(s)]
        n = s.size
        if n == 0:
            return np.nan
        if n == 1:
            return float(s[0])
        rho = self.rho_for(n)
        var = ((1.0 - rho) * np.sum(s ** 2) + rho * np.sum(s) ** 2) / n ** 2
        return float(np.sqrt(max(var, 0.0)))

    def __repr__(self):
        return f"CorrelationModel({self.name})"


INDEPENDENT = CorrelationModel(0.0, "independent")
FULLY_CORRELATED = CorrelationModel(1.0, "fully-correlated")


class UncertainTimeSeries(GenericTimeSeries):
    """
    A `~sunpy.timeseries.GenericTimeSeries` that carries typed uncertainty.

    Uncertainty is held in `uncertainty`, a mapping from data-column name to an
    `~astropy.nddata.NDUncertainty`, mirroring how `GenericTimeSeries` already
    holds `units` as a mapping from column name to `~astropy.units.Unit`. That
    parallel is deliberate: it is the smallest change that fits sunpy's existing
    shape, and it means uncertainty is not a row of the data table.

    Attributes
    ----------
    uncertainty : dict
        Column name -> `~astropy.nddata.NDUncertainty`.
    estimand : dict
        Column name -> one of `ESTIMANDS`. Records *what the sigma means*, which
        determines how it may legally be combined.
    """

    def __init__(self, data, meta=None, units=None, uncertainty=None,
                 estimand=None, **kwargs):
        super().__init__(data, meta=meta, units=units, **kwargs)
        self.uncertainty = dict(uncertainty or {})
        self.estimand = dict(estimand or {})
        for col, kind in self.estimand.items():
            if kind not in ESTIMANDS:
                raise ValueError(f"unknown estimand {kind!r} for {col!r}; "
                                 f"expected one of {sorted(ESTIMANDS)}")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_silso(cls, source: str = "daily_total", path=None) -> "UncertainTimeSeries":
        """Build from a SILSO product, attaching sigma as typed uncertainty."""
        from .baseline import SOURCES, build_meta

        frame, units = SOURCES[source](path)
        numeric = _numeric_only(frame)
        value_col = ("sunspot_number_smoothed" if "sunspot_number_smoothed" in numeric
                     else "sunspot_number")

        # sigma leaves the data table entirely and becomes typed uncertainty.
        data = numeric.drop(columns=["sigma"])
        unc = {value_col: StdDevUncertainty(numeric["sigma"].to_numpy(), unit=reader.SSN)}
        est = {value_col: "station_dispersion"}

        meta = dict(build_meta(source, frame))
        meta["uncertainty_estimand"] = est[value_col]
        meta["uncertainty_note"] = (
            "SILSO sigma is the standard deviation of the individual station "
            "counts, not the standard error of the mean. SILSO documents the "
            "standard error as sigma/sqrt(N)."
        )
        return cls(data, meta=meta,
                   units={k: v for k, v in units.items() if k in data.columns},
                   uncertainty=unc, estimand=est)

    # -- estimand conversion ----------------------------------------------
    def as_standard_error(self, column: str = "sunspot_number",
                          n_column: str = "n_observations") -> "UncertainTimeSeries":
        """
        Convert a station-dispersion uncertainty into a standard error of the mean.

        SILSO documents this explicitly: ``standard error = sigma / sqrt(N)``. This
        step is mandatory before any resampling, and it is the step that a plain
        "add a sigma column" design gives you no way to express, because nothing
        in that design records that the column was a dispersion to begin with.
        """
        if self.estimand.get(column) != "station_dispersion":
            raise ValueError(
                f"column {column!r} has estimand {self.estimand.get(column)!r}; "
                "as_standard_error() only applies to 'station_dispersion'.")

        frame = self.to_dataframe()
        sigma = np.asarray(self.uncertainty[column].array, dtype="float64")
        n = np.asarray(frame[n_column], dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = sigma / np.sqrt(n)

        new = self._clone(frame)
        new.uncertainty = dict(self.uncertainty)
        new.uncertainty[column] = StdDevUncertainty(sem, unit=reader.SSN)
        new.estimand = dict(self.estimand)
        new.estimand[column] = "standard_error"
        new.meta.metas[0]["uncertainty_estimand"] = "standard_error"
        return new

    # -- resampling --------------------------------------------------------
    def resample(self, rule: str, column: str = "sunspot_number",
                 correlation: CorrelationModel | float | None = None,
                 n_column: str = "n_observations") -> "UncertainTimeSeries":
        """
        Resample to a coarser cadence, propagating uncertainty explicitly.

        Parameters
        ----------
        rule : str
            A pandas offset alias, e.g. ``"MS"`` for calendar months.
        correlation : CorrelationModel or float
            **Required.** There is no defensible default; see `CorrelationModel`.

        Raises
        ------
        ValueError
            If ``correlation`` is not given, or if the column's uncertainty is
            still a station dispersion rather than a standard error.

        Notes
        -----
        Refusing to default ``correlation`` is the whole design claim. sunpy today
        has no resample at all on `~sunpy.timeseries.GenericTimeSeries`; if one is
        added and it silently averages a sigma column, it will produce numbers
        that are wrong by a factor of ~sqrt(N) without any warning.
        """
        if correlation is None:
            raise ValueError(
                "resample() requires an explicit `correlation`. Uncertainty "
                "propagation is undefined without it: rho=0 and rho=1 differ by "
                "a factor of sqrt(N) on this data. Pass INDEPENDENT, "
                "FULLY_CORRELATED, or a CorrelationModel.")
        if not isinstance(correlation, CorrelationModel):
            correlation = CorrelationModel(correlation)

        if self.estimand.get(column) == "station_dispersion":
            raise ValueError(
                f"column {column!r} still holds a station dispersion. Call "
                "as_standard_error() first. Averaging dispersions as if they "
                "were standard errors overstates the result by ~sqrt(N_stations).")

        frame = self.to_dataframe()
        sem = pd.Series(np.asarray(self.uncertainty[column].array, dtype="float64"),
                        index=frame.index)

        grouped = frame[column].resample(rule)
        values = grouped.mean()
        counts = grouped.count()
        combined = sem.resample(rule).apply(correlation.combine_mean)

        out = pd.DataFrame({column: values, "n_values": counts})
        if n_column in frame.columns:
            out[n_column] = frame[n_column].resample(rule).sum(min_count=1)

        meta = plain_meta(self)
        meta["resample_rule"] = rule
        meta["resample_correlation"] = correlation.name
        units = {k: v for k, v in self.units.items() if k in out.columns}
        units["n_values"] = reader.COUNT

        return type(self)(out, meta=meta, units=units,
                          uncertainty={column: StdDevUncertainty(combined.to_numpy(),
                                                                 unit=reader.SSN)},
                          estimand={column: "standard_error"})

    # -- helpers -----------------------------------------------------------
    def _clone(self, frame) -> "UncertainTimeSeries":
        return type(self)(frame, meta=plain_meta(self), units=dict(self.units))

    def sigma(self, column: str = "sunspot_number") -> pd.Series:
        """The uncertainty on ``column`` as a pandas Series aligned to the index."""
        return pd.Series(np.asarray(self.uncertainty[column].array, dtype="float64"),
                         index=self.to_dataframe().index, name=f"{column}_sigma")

    def __repr__(self):
        cols = ", ".join(f"{c}({self.estimand.get(c, '?')})" for c in self.uncertainty)
        return (f"<{type(self).__name__} {self.shape} "
                f"uncertainty on: {cols or 'none'}>")
