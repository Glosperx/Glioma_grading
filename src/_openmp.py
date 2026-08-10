"""Import-order guard for macOS.

On Apple Silicon both ``xgboost`` and ``torch`` ship their own OpenMP runtime. If torch is
imported first, xgboost segfaults at ``fit`` time rather than at import, which looks like a
data problem instead of a linking one. Importing xgboost first avoids it.

Any module that imports torch does ``from . import _openmp`` first, so the ordering holds
whichever module a notebook reaches for.
"""
try:  # pragma: no cover - environments without xgboost still work for the torch paths
    import xgboost  # noqa: F401
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False
