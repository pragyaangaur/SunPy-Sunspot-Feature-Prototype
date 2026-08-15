"""
What survives a GenericTimeSeries operation?

Run:  python scripts/api_audit.py

For every public operation that returns a *new* series, this checks four things:

  subclass     : is the returned object still the subclass it started as?
  units        : is the units dict intact for the columns that remain?
  meta         : are the metadata keys still reachable?
  uncertainty  : does the typed uncertainty attribute survive?

The point of the last column is that it is a proxy for "any state a subclass adds".
If a subclass cannot carry one extra attribute through an operation, then
attaching uncertainty to a time series is not an isolated feature request, it
needs an extension point that does not currently exist.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sunpy  # noqa: E402
from sunpy.timeseries import GenericTimeSeries  # noqa: E402

from silso_uncertainty.baseline import load, plain_meta  # noqa: E402
from silso_uncertainty.uncertain import (INDEPENDENT, UncertainTimeSeries)  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OK, NO, NA = "yes", "NO", "n/a"


def _units_ok(before, after) -> str:
    """Units preserved for every column that still exists?"""
    try:
        kept = [c for c in after.columns if c in before.units]
        if not kept:
            return NA
        return OK if all(after.units.get(c) == before.units[c] for c in kept) else NO
    except Exception:
        return NO


def _meta_ok(before, after) -> str:
    """Are the original metadata *keys and values* still reachable?

    Note this deliberately does not use ``TimeSeriesMetaData.get``, which returns
    a pretty-printed view whose header embeds the time range, so comparing those
    strings reports a spurious loss for every operation that changes the range.
    The comparison is on the merged MetaDict contents instead.
    """
    try:
        want = plain_meta(before)
        got = plain_meta(after)
        missing = [k for k, v in want.items() if got.get(k) != v]
        return OK if not missing else f"NO ({len(missing)} keys)"
    except Exception:
        return NO


def _unc_ok(before, after) -> str:
    if not isinstance(before, UncertainTimeSeries):
        return NA
    if not isinstance(after, UncertainTimeSeries):
        return NO  # subclass lost, so the attribute is gone with it
    return OK if getattr(after, "uncertainty", None) else NO


def run_case(name: str, before, op) -> dict:
    """Apply ``op`` to ``before`` and record what survived."""
    row = {"operation": name, "subclass": NO, "units": NO, "meta": NO,
           "uncertainty": NO, "note": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            after = op(before)
    except Exception as exc:
        row.update({"subclass": NA, "units": NA, "meta": NA, "uncertainty": NA,
                    "note": f"raised {type(exc).__name__}: {str(exc)[:60]}"})
        return row

    row["subclass"] = OK if type(after) is type(before) else NO
    if type(after) is not type(before):
        row["note"] = f"-> {type(after).__name__}"
    row["units"] = _units_ok(before, after)
    row["meta"] = _meta_ok(before, after)
    row["uncertainty"] = _unc_ok(before, after)
    return row


def build_cases(series):
    """The operations under test, as (name, callable) pairs."""
    idx = series.to_dataframe().index
    mid = idx[len(idx) // 2]

    def _concat(s):
        a = s.truncate(str(idx[0].date()), str(mid.date()))
        b = s.truncate(str(mid.date()), str(idx[-1].date()))
        return a.concatenate(b)

    def _concat_same_source(s):
        a = s.truncate(str(idx[0].date()), str(mid.date()))
        b = s.truncate(str(mid.date()), str(idx[-1].date()))
        return a.concatenate(b, same_source=True)

    def _add(s):
        n = len(s.to_dataframe())
        return s.add_column("scratch", u.Quantity(np.ones(n), u.dimensionless_unscaled))

    return [
        ("truncate", lambda s: s.truncate(str(idx[0].date()), str(mid.date()))),
        ("concatenate", _concat),
        ("concatenate(same_source=True)", _concat_same_source),
        ("sort_index", lambda s: s.sort_index()),
        ("add_column", _add),
        ("remove_column", lambda s: s.remove_column("year_fraction")),
        ("extract", lambda s: s.extract("sunspot_number")),
        ("resample", lambda s: s.resample("MS")),
    ]


def table(rows: list[dict]) -> str:
    cols = ["operation", "subclass", "units", "meta", "uncertainty", "note"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    head = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    rule = "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    body = ["| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, rule] + body)


def main() -> None:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"sunpy version under test: {sunpy.__version__}")
    emit(f"GenericTimeSeries.resample exists: {hasattr(GenericTimeSeries, 'resample')}")
    emit()

    plain = load("daily_total").truncate("1900-01-01", "1910-01-01")
    unc = UncertainTimeSeries.from_silso("daily_total").truncate(
        "1900-01-01", "1910-01-01")
    # truncate may already have downcast; rebuild the subclass instance directly.
    if not isinstance(unc, UncertainTimeSeries) or not unc.uncertainty:
        full = UncertainTimeSeries.from_silso("daily_total")
        df = full.to_dataframe().loc["1900-01-01":"1910-01-01"]
        sig = full.sigma().loc["1900-01-01":"1910-01-01"]
        from astropy.nddata import StdDevUncertainty

        unc = UncertainTimeSeries(
            df, meta=plain_meta(full), units=dict(full.units),
            uncertainty={"sunspot_number": StdDevUncertainty(sig.to_numpy())},
            estimand={"sunspot_number": "station_dispersion"})

    for label, series in [("GenericTimeSeries (baseline)", plain),
                          ("UncertainTimeSeries (prototype)", unc)]:
        emit("=" * 92)
        emit(label)
        emit("=" * 92)
        rows = [run_case(name, series, op) for name, op in build_cases(series)]
        emit(table(rows))
        emit()

    emit("=" * 92)
    emit("How each operation constructs its return value (sunpy/timeseries/timeseriesbase.py)")
    emit("=" * 92)
    emit("  add_column      self.__class__(data, meta, units)")
    emit("  remove_column   self.__class__(data, deepcopy(meta), units)")
    emit("  truncate        self.__class__(data.sort_index(), meta, copy(units))")
    emit("  concatenate     self.__class__(...) if all operands share a class,")
    emit("                  else GenericTimeSeries(...)   [NOT switched on same_source]")
    emit("  sort_index      GenericTimeSeries(...)   <- hardcoded, downcasts")
    emit("  extract         GenericTimeSeries(...)   <- hardcoded, downcasts")
    emit()
    emit("Every one of these calls the three-argument constructor positionally.")
    emit("There is no _new_instance / _replace hook a subclass can override, so a")
    emit("subclass carrying extra state loses it even when its type is preserved.")

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "api_audit.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
