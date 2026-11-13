"""Make sure every package the notebooks and the app need is importable.

Called from the first cell of each notebook and from ``app.py``. Only packages
that are missing (or broken) are installed, so an existing setup such as
Google Colab's own OpenCV or NumPy is never replaced. Works with plain Python,
virtual environments, conda, Colab, Kaggle and VS Code / Jupyter on Windows,
macOS and Linux.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

# import name -> pip requirement
REQUIREMENTS = {
    "numpy": "numpy>=1.24",
    "pandas": "pandas>=2.0",
    "matplotlib": "matplotlib>=3.6",
    "seaborn": "seaborn>=0.12",
    "sklearn": "scikit-learn>=1.3",
    "skimage": "scikit-image>=0.21",
    "cv2": "opencv-python-headless>=4.8,<5",
    "librosa": "librosa>=0.10",
    "soundfile": "soundfile>=0.12",
    "PIL": "Pillow>=10",
    "pillow_heif": "pillow-heif>=0.16",
    "av": "av>=12",
    "xgboost": "xgboost>=2.0",
    "joblib": "joblib>=1.3",
}


def _usable(module: str) -> bool:
    if importlib.util.find_spec(module) is None:
        return False
    if module == "cv2":  # the face detector files must ship with OpenCV
        try:
            import cv2
            from pathlib import Path

            return Path(cv2.data.haarcascades, "haarcascade_frontalface_default.xml").exists()
        except Exception:  # noqa: BLE001
            return False
    return True


def missing_packages() -> list[str]:
    """pip requirements whose import name is not available in this environment."""
    return [req for module, req in REQUIREMENTS.items() if not _usable(module)]


def ensure_packages(quiet: bool = True) -> None:
    """Install the missing requirements into the running interpreter."""
    missing = missing_packages()
    if not missing:
        print("All required packages are available.")
        return
    print("Installing missing packages:", ", ".join(missing))
    cmd = [sys.executable, "-m", "pip", "install", *(["-q"] if quiet else []), *missing]
    result = subprocess.run(cmd)
    if result.returncode != 0:  # e.g. a system Python that refuses global installs
        result = subprocess.run(cmd + ["--user"])
    importlib.invalidate_caches()
    still_missing = missing_packages()
    if still_missing:
        raise ImportError(
            "Could not install: " + ", ".join(still_missing)
            + ". Run `pip install -r requirements.txt` in a terminal, then restart the kernel."
        )
    print("Done.")


if __name__ == "__main__":
    ensure_packages(quiet=False)
