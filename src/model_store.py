"""Save and load trained models so they work on any machine.

A pickled scikit-learn model only loads reliably with the same scikit-learn
version that saved it. Every model therefore records the version it was built
with; when the file is missing, cannot be read, or was built with another
version, it is simply retrained from the CSV files in ``datasets/`` (a few
seconds), so the app never fails because of a different environment.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import joblib
import sklearn


def save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.sklearn_version = sklearn.__version__
    joblib.dump(model, path, compress=3)


def load_or_train(path: Path, train: Callable[..., object], verbose: bool = False):
    """Load ``path``; retrain with ``train`` (and save) if it is unusable here."""
    reason = None
    if not path.exists():
        reason = "not trained yet"
    else:
        try:
            with warnings.catch_warnings():  # a version mismatch is handled below by retraining
                warnings.simplefilter("ignore")
                model = joblib.load(path)
            if getattr(model, "sklearn_version", None) == sklearn.__version__:
                return model
            reason = f"saved with scikit-learn {getattr(model, 'sklearn_version', '?')}, " \
                     f"this environment has {sklearn.__version__}"
        except Exception as error:  # noqa: BLE001 - any unreadable file is retrained
            reason = f"could not be loaded ({type(error).__name__})"
    print(f"{path.name}: {reason}; training it now from the feature files...")
    model = train(verbose=verbose)
    save(model, path)
    return model
