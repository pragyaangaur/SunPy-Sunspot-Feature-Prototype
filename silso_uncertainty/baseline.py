"""
SILSO data in a `~sunpy.timeseries.GenericTimeSeries`, built the way sunpy would
actually build it today.

This is deliberately the *honest* baseline. sunpy has no concept of uncertainty,
so sigma and the station count become ordinary numeric columns, indistinguishable
from the sunspot number itself. Nothing in the object records that ``sigma`` is an
uncertainty on ``sunspot_number`` rather than a second measurement.

That is precisely the gap this prototype is measuring, so the baseline does not
try to work around it.
"""
from __future__ import annotations

import pandas as pd
from sunpy.timeseries import GenericTimeSeries
from sunpy.util.metadata import MetaDict

from . import reader

#: Readers exposed through `load`.
SOURCES = {
    "daily_total": reader.read_daily_total,
    "daily_hemispheric": reader.read_daily_hemispheric,
    "monthly_total": reader.read_monthly_total,
    "monthly_smoothed": reader.read_monthly_smoothed,
    "yearly_total": reader.read_yearly_total,
}


def _numeric_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the boolean ``definitive`` flag.

    `GenericTimeSeries` stores its data in a DataFrame that is expected to be
    numeric throughout: ``to_array`` and the plotting code both assume it. The
    provisional/definitive flag is carried in ``meta`` instead so it is not lost.
    """
    return frame.drop(columns=[c for c in ("definitive",) if c in frame.columns])


def build_meta(source: str, frame: pd.DataFrame) -> MetaDict:
    """The metadata block attached to every SILSO series built here."""
    return MetaDict({
        "instrume": "SILSO",
        "observatory": "SILSO",
        "telescop": "SILSO",
        "source": "SILSO",
        "detector": source,
        "silso_product": source,
        "comment": ("Sunspot Number version 2.0, WDC-SILSO, Royal Observatory of "
                    "Belgium, Brussels. Values flagged provisional are subject to "
                    "revision."),
        "n_provisional_rows": int((~frame["definitive"]).sum()),
    })


def plain_meta(timeseries) -> dict:
    """
    Flatten a `~sunpy.timeseries.metadata.TimeSeriesMetaData` back to a dict.

    `GenericTimeSeries` wraps whatever ``meta`` it is handed in a
    `TimeSeriesMetaData`, which is a list of (time range, columns, MetaDict)
    triples and is not itself dict-like. Round-tripping metadata through a
    subclass constructor therefore needs this unwrap step, itself a small piece
    of evidence about how much of the object survives an operation.
    """
    merged = {}
    for entry in getattr(timeseries.meta, "metas", [timeseries.meta]):
        merged.update(dict(entry))
    return merged


def load(source: str = "daily_total", path=None) -> GenericTimeSeries:
    """
    Build a plain `~sunpy.timeseries.GenericTimeSeries` for a SILSO product.

    Parameters
    ----------
    source : str
        One of the keys of `SOURCES`.

    Returns
    -------
    sunpy.timeseries.GenericTimeSeries
        With ``sigma`` and ``n_observations`` as ordinary untyped columns.
    """
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; known: {sorted(SOURCES)}")

    frame, units = SOURCES[source](path)
    meta = build_meta(source, frame)
    numeric = _numeric_only(frame)
    return GenericTimeSeries(numeric, meta=meta, units={k: v for k, v in units.items()
                                                        if k in numeric.columns})
