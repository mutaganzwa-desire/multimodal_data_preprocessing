"""Facial recognition model: the first security check.

Trains on ``datasets/image_features.csv`` (532 colour-histogram, LBP and HOG
features per face crop), evaluates with leave-one-photo-out cross-validation
(accuracy, macro F1, log loss) and saves the model to ``models/face_model.joblib``.

    python -m src.face_model            # train, evaluate and save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import image_pipeline as ip
from src.model_store import load_or_train, save
from src.verification import BiometricModel, Decision

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "face_model.joblib"


def train(features_csv: Path = ip.OUTPUT_CSV, exclude: tuple[str, ...] = (),
          verbose: bool = True, reuse: BiometricModel | None = None) -> BiometricModel:
    """Train the face model.

    ``exclude`` leaves members out and ``reuse`` keeps an existing model's
    thresholds; together they simulate a stranger in the demo.
    """
    table = pd.read_csv(features_csv)
    table = table[~table["member"].isin(exclude)].reset_index(drop=True)
    # The model learns identity, so every member must have at least two photos.
    return BiometricModel.train(table, ip.feature_columns(table), "source_image",
                                kind="face", outsider_claim="argmax", verbose=verbose, reuse=reuse)


def load(path: Path = MODEL_PATH) -> BiometricModel:
    """Load the saved model, retraining it if it is missing or incompatible."""
    return load_or_train(path, train)


def recognise(model: BiometricModel, image_path: Path | str) -> Decision:
    """Run the face check on one photo of any format."""
    face, info = ip.load_face(image_path)
    if not info["face_detected"]:
        return Decision(None, 0.0, -1.0, False, "no face detected in the photo")
    return model.identify(ip.extract_features(face))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    model = train()
    save(model, args.output)
    m = model.metrics["closed_set"]
    print(f"Saved {args.output.relative_to(REPO_ROOT)} | accuracy {m['accuracy']:.3f}, "
          f"F1 {m['f1_macro']:.3f}, log loss {m['log_loss']:.3f}")


if __name__ == "__main__":
    main()
