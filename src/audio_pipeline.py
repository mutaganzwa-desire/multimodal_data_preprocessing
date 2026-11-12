"""Voice sample pipeline: load, clean, augment, and extract features.

The same functions build ``datasets/audio_features.csv`` and, later, turn one
new recording into a feature vector for the voiceprint verification model, so
training and inference always see identical preprocessing.

Run from the repository root to regenerate the feature file:

    python -m src.audio_pipeline
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data" / "audio"
OUTPUT_CSV = REPO_ROOT / "datasets" / "audio_features.csv"

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm"}
SAMPLE_RATE = 16_000     # plenty for speech, and the same for every device
TRIM_TOP_DB = 30         # anything 30 dB below the loudest part counts as silence
MAX_WORD_GAP_S = 0.4     # pauses shorter than this are gaps between words
PEAK_LEVEL = 0.95
N_MFCC = 13
NOISE_SNR_DB = 20
SEED = 42


# ---------------------------------------------------------------- loading ---

def find_clips(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """List every voice clip as (member, phrase, take, path).

    Clips live in ``data/audio/<member>/<member>_<phrase>_<take>.<ext>``. The
    folder name is the speaker label.
    """
    rows = []
    for path in sorted(audio_dir.glob("*/*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        member = path.parent.name
        name = path.stem.removeprefix(f"{member}_")
        phrase, _, take = name.rpartition("_")
        if not take.isdigit():
            phrase, take = name, "1"
        rows.append({"member": member, "phrase": phrase, "take": int(take), "path": path})
    return pd.DataFrame(rows)


def _decode_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg is needed to read {path.suffix} files such as {path}")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_audio(path: Path | str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any clip as mono float32 at ``sr`` Hz.

    WAV, FLAC and OGG go through soundfile; phone formats such as AAC and M4A
    are decoded with ffmpeg (preinstalled on Colab).
    """
    path = Path(path)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        return _decode_with_ffmpeg(path, sr)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def speech_interval(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, int]:
    """Return (start, end) sample indices of the spoken phrase.

    Sound regions are found with ``librosa.effects.split``. Regions separated by
    a short pause (the gap between two words) are joined, and the joined region
    with the most energy is kept. A plain trim would instead keep everything up
    to a faint click a second after the phrase.
    """
    intervals = librosa.effects.split(y, top_db=TRIM_TOP_DB)
    if len(intervals) == 0:
        return 0, len(y)
    max_gap = int(MAX_WORD_GAP_S * sr)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    start, end = max(merged, key=lambda se: np.sum(y[se[0]:se[1]] ** 2))
    return int(start), int(end)


def trim_to_speech(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Keep only the spoken phrase, dropping silence and stray clicks around it."""
    start, end = speech_interval(y, sr)
    return y[start:end]


def clean(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cut the clip down to the phrase, then normalise the peak level.

    The two members recorded on different phones with different amounts of
    silence and loudness. Without this step a model could tell speakers apart
    by clip length or volume instead of by their voice.
    """
    y = trim_to_speech(y, sr)
    peak = np.max(np.abs(y))
    return y * (PEAK_LEVEL / peak) if peak > 0 else y


def load_clip(path: Path | str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and clean one clip: the input to augmentation and feature extraction."""
    return clean(load_audio(path, sr), sr)


# ----------------------------------------------------------- augmentation ---

def add_background_noise(y: np.ndarray, snr_db: float = NOISE_SNR_DB,
                         seed: int = SEED) -> np.ndarray:
    """Mix in white noise at a fixed signal-to-noise ratio (a noisy room)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy = y + rng.normal(0, np.sqrt(noise_power), y.shape).astype(np.float32)
    return np.clip(noisy, -1.0, 1.0)


def change_gain(y: np.ndarray, db: float) -> np.ndarray:
    """Make the clip quieter or louder (speaking further from the microphone)."""
    return np.clip(y * (10 ** (db / 20)), -1.0, 1.0)


# Each augmentation imitates a way the same person may sound at login time:
# a slightly higher or lower voice, speaking faster or slower, background noise,
# or holding the phone further away.
AUGMENTATIONS = {
    "original": lambda y, sr: y,
    "pitch_up_2st": lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
    "pitch_down_2st": lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
    "time_stretch_fast": lambda y, sr: librosa.effects.time_stretch(y, rate=1.15),
    "time_stretch_slow": lambda y, sr: librosa.effects.time_stretch(y, rate=0.85),
    "background_noise": lambda y, sr: add_background_noise(y),
    "gain_down_6db": lambda y, sr: change_gain(y, -6),
}


def augment(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Return every augmented version of a cleaned clip, keyed by name."""
    return {name: fn(y, sr).astype(np.float32) for name, fn in AUGMENTATIONS.items()}


# ------------------------------------------------------------- features ---

def _mean_std(name: str, values: np.ndarray) -> dict[str, float]:
    return {f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))}


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Summarise a clip as fixed-length features (frame statistics over time).

    * MFCC 1-13 mean and std: the shape of the vocal tract, the core voiceprint.
    * MFCC delta std: how quickly that shape changes while speaking.
    * Spectral roll-off, centroid and bandwidth: where the energy sits in
      frequency (brighter or darker voice).
    * RMS energy: loudness and its variation (speech rhythm).
    * Zero-crossing rate: how noisy or breathy the sound is.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    feats: dict[str, float] = {"duration_s": len(y) / sr}
    for i in range(N_MFCC):
        feats.update(_mean_std(f"mfcc_{i + 1}", mfcc[i]))
    for i in range(N_MFCC):
        feats[f"mfcc_delta_{i + 1}_std"] = float(np.std(delta[i]))
    feats.update(_mean_std("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats.update(_mean_std("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats.update(_mean_std("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats.update(_mean_std("rms_energy", librosa.feature.rms(y=y)))
    feats.update(_mean_std("zcr", librosa.feature.zero_crossing_rate(y)))
    return feats


def features_for_path(path: Path | str) -> dict[str, float]:
    """Features for one new recording, as used at login time (no augmentation)."""
    return extract_features(load_clip(path))


# ---------------------------------------------------------------- table ---

METADATA_COLUMNS = ["clip_id", "member", "phrase", "take", "augmentation", "source_clip"]


def build_feature_table(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """One row per (clip, augmentation) with metadata columns, then features."""
    rows = []
    for item in find_clips(audio_dir).itertuples():
        y = load_clip(item.path)
        for aug_name, aug_y in augment(y).items():
            rows.append({
                "clip_id": f"{item.path.stem}__{aug_name}",
                "member": item.member,
                "phrase": item.phrase,
                "take": item.take,
                "augmentation": aug_name,
                "source_clip": item.path.relative_to(audio_dir.parent.parent).as_posix(),
                **extract_features(aug_y),
            })
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The numeric feature columns of an audio feature table (no metadata)."""
    return [c for c in table.columns if c not in METADATA_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    table = build_feature_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")
    n_feat = len(feature_columns(table))
    print(f"Wrote {len(table)} rows x {n_feat} features to {args.output}")
    print(table.groupby(["member", "phrase"]).size().to_string())


if __name__ == "__main__":
    main()
