"""Project package.

Compatibility guard, run before any module imports librosa:

librosa resamples audio with ``soxr``, a compiled library. Some managed Windows
PCs (Smart App Control / Windows Defender Application Control) block its DLL,
which makes every resample, pitch shift and time stretch crash with
"DLL load failed while importing soxr_ext". When ``soxr`` cannot be loaded, a
small stand-in module is registered that resamples with SciPy instead, so the
pipelines, notebooks and app keep working. Results differ only in the last
decimal places.
"""

import sys
import types


def _install_soxr_fallback() -> None:
    try:
        import soxr  # noqa: F401
        return
    except Exception:  # ImportError, OSError: missing or blocked by policy
        pass

    from fractions import Fraction

    def resample(x, in_rate, out_rate, quality="HQ"):  # same signature as soxr.resample
        import numpy as np  # imported here: the packages may not be installed yet at startup
        from scipy.signal import resample_poly

        x = np.asarray(x)
        ratio = Fraction(int(round(out_rate)), int(round(in_rate))).limit_denominator(1000)
        return resample_poly(x, ratio.numerator, ratio.denominator, axis=0).astype(x.dtype, copy=False)

    stub = types.ModuleType("soxr")
    stub.resample = resample
    stub.__version__ = "scipy-fallback"
    stub.__doc__ = "Stand-in for soxr (blocked or missing): resampling via scipy.signal.resample_poly."
    sys.modules["soxr"] = stub


_install_soxr_fallback()
