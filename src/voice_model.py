"""Voiceprint verification model: the check that approves a transaction.

Trains on ``datasets/audio_features.csv`` (50 MFCC, spectral, energy and rhythm
features per clip), evaluates with leave-one-recording-out cross-validation
(accuracy, macro F1, log loss) and saves the model to ``models/voice_model.joblib``.

At login the voice must match the member that the face model recognised, so a
teammate cannot approve someone else's transaction and a stranger is rejected.

    python -m src.voice_model           # train, evaluate and save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import audio_pipeline as ap
from src.model_store import load_or_train, save
from src.verification import BiometricModel, Decision

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "voice_model.joblib"


def train(features_csv: Path = ap.OUTPUT_CSV, exclude: tuple[str, ...] = (),
          verbose: bool = True, reuse: BiometricModel | None = None) -> BiometricModel:
    """Train the voice model.

    ``exclude`` leaves members out and ``reuse`` keeps an existing model's
    thresholds; together they simulate a stranger in the demo.
    """
    table = pd.read_csv(features_csv)
    table = table[~table["member"].isin(exclude)].reset_index(drop=True)
    return BiometricModel.train(table, ap.feature_columns(table), "source_clip",
                                kind="voice", outsider_claim="all", verbose=verbose, reuse=reuse)


def load(path: Path = MODEL_PATH) -> BiometricModel:
    """Load the saved model, retraining it if it is missing or incompatible."""
    return load_or_train(path, train)


def verify(model: BiometricModel, audio_path: Path | str, claimed_member: str) -> Decision:
    """Run the voice check on one recording of any format, against the face result."""
    return model.verify(ap.features_for_path(audio_path), claimed_member)


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
