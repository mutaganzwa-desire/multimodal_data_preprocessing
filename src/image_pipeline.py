"""Face image pipeline: load, crop the face, augment, and extract features.

The same functions are used to build ``datasets/image_features.csv`` and, later,
to turn a single new photo into a feature vector for the face recognition model,
so training and inference always see identical preprocessing.

Run from the repository root to regenerate the feature file:

    python -m src.image_pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = REPO_ROOT / "data" / "images"
OUTPUT_CSV = REPO_ROOT / "datasets" / "image_features.csv"

EXPRESSIONS = ("neutral", "smiling", "surprised")
FACE_SIZE = 128          # side of the square face crop, in pixels
FACE_MARGIN = 0.15       # extra border kept around the detected face box
HIST_BINS = 16           # bins per colour channel
LBP_POINTS, LBP_RADIUS, LBP_GRID = 8, 1, 4
HOG_SIZE, HOG_CELL = 64, 16
WORK_SIDE = 800         # photos are shrunk to this longest side before detection
ROTATION_SEARCH = (0, -10, 10, -20, 20, -30, 30)
SEED = 42

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ---------------------------------------------------------------- loading ---

def find_images(image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    """List every member photo as (member, expression, path).

    Photos live in ``data/images/<member>/<member>_<expression>.jpg``. The folder
    name is the identity label; the expression comes from the file name.
    """
    rows = []
    for path in sorted(image_dir.glob("*/*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        member = path.parent.name
        expression = path.stem.removeprefix(f"{member}_").split("_")[0]
        rows.append({"member": member, "expression": expression, "path": path})
    return pd.DataFrame(rows)


def load_image(path: Path | str) -> np.ndarray:
    """Read a photo as an RGB uint8 array, applying the phone's EXIF rotation."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def _rotate_full(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate without cutting corners off (the canvas grows to fit)."""
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)


def _count_eyes(gray: np.ndarray, box) -> int:
    """Count eyes in the upper part of a face box, to reject false face hits."""
    x, y, w, h = box
    upper = gray[y:y + int(h * 0.6), x:x + w]
    eyes = _EYE_CASCADE.detectMultiScale(
        upper, scaleFactor=1.1, minNeighbors=5,
        minSize=(w // 10, w // 10), maxSize=(w // 3, w // 3),
    )
    return len(eyes)


def locate_face(rgb: np.ndarray) -> tuple[np.ndarray, tuple | None, int]:
    """Find the face, straightening a tilted head first if needed.

    The Haar detector only finds upright faces, so the photo is also tried
    rotated by up to 30 degrees either way. A candidate counts only if eyes are
    found in its upper half, and the smallest rotation wins. Returns the
    (possibly rotated) image, the face box (x, y, w, h) or None, and the angle.
    """
    scale = WORK_SIDE / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    candidates = []
    for angle in ROTATION_SEARCH:
        image = rgb if angle == 0 else _rotate_full(rgb, angle)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        min_side = max(30, min(gray.shape) // 6)
        faces, _, weights = _FACE_CASCADE.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_side, min_side), outputRejectLevels=True,
        )
        for box, weight in zip(faces, np.ravel(weights)):
            box = tuple(int(v) for v in box)
            eyes = min(_count_eyes(gray, box), 2)
            # Rank: more eyes found, then less rotation, then detector confidence.
            candidates.append(((-eyes, abs(angle), -float(weight)), image, box, angle))

    if not candidates:
        return rgb, None, 0
    _, image, box, angle = min(candidates, key=lambda c: c[0])
    return image, box, angle


def crop_face(rgb: np.ndarray, box=None, size: int = FACE_SIZE) -> np.ndarray:
    """Crop a square around the face (with a small margin) and resize it.

    Falls back to a centred square crop when no face box is available, so the
    pipeline never fails on an unusual photo.
    """
    h_img, w_img = rgb.shape[:2]
    if box is None:
        side = min(h_img, w_img)
        x0, y0 = (w_img - side) // 2, (h_img - side) // 2
    else:
        x, y, w, h = box
        side = int(max(w, h) * (1 + 2 * FACE_MARGIN))
        side = min(side, h_img, w_img)
        cx, cy = x + w // 2, y + h // 2
        x0 = int(np.clip(cx - side // 2, 0, w_img - side))
        y0 = int(np.clip(cy - side // 2, 0, h_img - side))
    crop = rgb[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def load_face(path: Path | str) -> tuple[np.ndarray, dict]:
    """Load a photo and return (face crop, detection info)."""
    image, box, angle = locate_face(load_image(path))
    return crop_face(image, box), {"face_detected": box is not None, "face_angle": angle}


# ----------------------------------------------------------- augmentation ---

def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _gaussian_noise(img: np.ndarray, sigma: float = 12.0, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Each augmentation imitates a condition the face model may meet at login time:
# a tilted head, a mirrored front camera, a black and white or badly lit photo,
# or a grainy low-light sensor.
AUGMENTATIONS = {
    "original": lambda img: img,
    "rotate_+15": lambda img: _rotate(img, 15),
    "rotate_-15": lambda img: _rotate(img, -15),
    "flip_horizontal": lambda img: cv2.flip(img, 1),
    "grayscale": _grayscale,
    "brightness_up": lambda img: _brightness(img, 1.4),
    "brightness_down": lambda img: _brightness(img, 0.6),
    "gaussian_noise": _gaussian_noise,
}


def augment(face: np.ndarray) -> dict[str, np.ndarray]:
    """Return every augmented version of a face crop, keyed by augmentation name."""
    return {name: fn(face) for name, fn in AUGMENTATIONS.items()}


# ------------------------------------------------------------- features ---

def color_histogram(face: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Normalised per-channel RGB histogram (3 x bins values): skin tone and lighting."""
    hists = [
        np.histogram(face[..., c], bins=bins, range=(0, 256))[0] for c in range(3)
    ]
    hist = np.concatenate(hists).astype(np.float64)
    return hist / hist.sum()


def lbp_histogram(face: np.ndarray) -> np.ndarray:
    """Uniform LBP histograms over a 4 x 4 grid: local skin and facial texture."""
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    cell = gray.shape[0] // LBP_GRID
    feats = []
    for row in range(LBP_GRID):
        for col in range(LBP_GRID):
            patch = lbp[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
            hist = np.histogram(patch, bins=n_bins, range=(0, n_bins))[0]
            feats.append(hist / max(hist.sum(), 1))
    return np.concatenate(feats)


def hog_descriptor(face: np.ndarray) -> np.ndarray:
    """HOG on a 64 x 64 grey face: the shape of the eyes, nose, mouth and jaw."""
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(
        small,
        orientations=9,
        pixels_per_cell=(HOG_CELL, HOG_CELL),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def extract_features(face: np.ndarray) -> dict[str, float]:
    """Turn one face crop into a flat, named feature dictionary."""
    blocks = {
        "hist": color_histogram(face),
        "lbp": lbp_histogram(face),
        "hog": hog_descriptor(face),
    }
    return {
        f"{prefix}_{i}": float(value)
        for prefix, values in blocks.items()
        for i, value in enumerate(values)
    }


def features_for_path(path: Path | str) -> dict[str, float]:
    """Features for one new photo, as used at login time (no augmentation)."""
    face, _ = load_face(path)
    return extract_features(face)


# ---------------------------------------------------------------- table ---

def build_feature_table(image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    """One row per (photo, augmentation) with metadata columns, then features."""
    rows = []
    for item in find_images(image_dir).itertuples():
        face, info = load_face(item.path)
        for aug_name, aug_face in augment(face).items():
            rows.append({
                "image_id": f"{item.path.stem}__{aug_name}",
                "member": item.member,
                "expression": item.expression,
                "augmentation": aug_name,
                "source_image": item.path.relative_to(image_dir.parent.parent).as_posix(),
                **info,
                **extract_features(aug_face),
            })
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The numeric feature columns of an image feature table (no metadata)."""
    return [c for c in table.columns if c.split("_")[0] in {"hist", "lbp", "hog"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    table = build_feature_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")
    n_feat = len(feature_columns(table))
    print(f"Wrote {len(table)} rows x {n_feat} features to {args.output}")
    print(table.groupby(["member", "expression"]).size().to_string())


if __name__ == "__main__":
    main()
