"""
A SILSO sunspot-number reader with a first-class uncertainty channel.

An out-of-tree prototype for sunpy issue #8750: what it would take for
`sunpy.timeseries.GenericTimeSeries` to represent uncertainty. See the README and
the README for the findings.
"""
from .baseline import load
from .uncertain import (FULLY_CORRELATED, INDEPENDENT, CorrelationModel,
                        UncertainTimeSeries)

__all__ = ["load", "UncertainTimeSeries", "CorrelationModel",
           "INDEPENDENT", "FULLY_CORRELATED"]
