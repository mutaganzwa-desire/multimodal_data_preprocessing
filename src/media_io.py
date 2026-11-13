"""Read any image or audio file, whatever its format or extension.

Members record on different phones, so the raw files arrive as JPEG, PNG, HEIC
(iPhone), WEBP, WAV, M4A, AAC, MP3, OGG, OPUS and so on. Sometimes the extension
is even wrong (an M4A renamed to ``.wav``). Everything in this module looks at
the file's *content*, not its name:

* Images are opened with Pillow (plus ``pillow-heif`` for HEIC/HEIF/AVIF), with
  OpenCV as a last resort, and always returned as RGB with the phone's EXIF
  rotation applied.
* Audio is read with ``soundfile`` (WAV, FLAC, OGG, AIFF, recent MP3) and falls
  back to PyAV, which ships its own FFmpeg inside the pip wheel, so M4A, AAC,
  MP4, WEBM, OPUS, AMR, 3GP and WMA work on Windows, macOS, Linux and Colab
  without installing any system program.

``python -m src.media_io`` converts the files in ``data/`` to the canonical
formats (16-bit mono WAV and JPEG) and names, so the repository stays tidy.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".heic", ".heif", ".avif",
}
AUDIO_EXTENSIONS = {
    ".wav", ".wave", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".mp3", ".m4a",
    ".aac", ".mp4", ".webm", ".opus", ".amr", ".3gp", ".3gpp", ".wma", ".caf",
}

# Words people use in file names, mapped to the labels the pipelines expect.
EXPRESSION_SYNONYMS = {
    "neutral": "neutral", "nuetral": "neutral", "neutal": "neutral", "normal": "neutral", "plain": "neutral",
    "smiling": "smiling", "smile": "smiling", "smiles": "smiling", "happy": "smiling",
    "surprised": "surprised", "surprise": "surprised", "shocked": "surprised",
}
PHRASE_KEYWORDS = {"approve": "yes_approve", "confirm": "confirm_transaction"}

try:  # HEIC / HEIF (iPhone) and AVIF support for Pillow, when installed.
    import pillow_heif

    pillow_heif.register_heif_opener()
    if hasattr(pillow_heif, "register_avif_opener"):
        pillow_heif.register_avif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pillow_heif = None


# ------------------------------------------------------------------ names ---

def warn(message: str) -> None:
    """Print a short warning (no internal file paths), visible in notebooks and terminals."""
    print(f"WARNING: {message}", file=sys.stderr)



def slug(text: str) -> str:
    """Lower-case a name and turn spaces, dashes and dots into underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_media_file(path: Path, kind: str) -> bool:
    """True for a visible file whose extension belongs to ``kind`` ('image' or 'audio')."""
    allowed = IMAGE_EXTENSIONS if kind == "image" else AUDIO_EXTENSIONS
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in allowed


def expression_from_name(stem: str, member: str = "") -> str:
    """Find the expression in a file name such as ``Ivan_Smile`` or ``arnold-neutral``."""
    words = slug(stem).removeprefix(slug(member) + "_").split("_")
    for word in words:
        if word in EXPRESSION_SYNONYMS:
            return EXPRESSION_SYNONYMS[word]
    return words[0] if words and words[0] else "unknown"


def phrase_and_take(stem: str, member: str = "") -> tuple[str, int]:
    """Split ``alain_yes_approve_2`` into ('yes_approve', 2). Missing take -> 1."""
    name = slug(stem).removeprefix(slug(member) + "_")
    match = re.search(r"_?(\d+)$", name)
    take = int(match.group(1)) if match else 1
    if match:
        name = name[: match.start()]
    for keyword, phrase in PHRASE_KEYWORDS.items():
        if keyword in name:
            return phrase, take
    return name or "unknown", take


def report_loose_files(folder: Path, kind: str) -> list[Path]:
    """Warn about media files placed directly in ``folder`` instead of a member folder."""
    loose = [p for p in sorted(folder.glob("*")) if is_media_file(p, kind)]
    if loose:
        names = ", ".join(p.name for p in loose)
        warn(f"{len(loose)} {kind} file(s) in {folder.name}/ are not inside a member folder and "
             f"were skipped: {names}. Move them to {folder.name}/<member>/.")
    return loose


def drop_duplicate_files(table, kind: str):
    """Remove rows whose file is a byte-for-byte copy of an earlier row's file.

    A copied file (for example the same take saved twice as ``_1`` and ``_2``)
    adds no new information, and in a grouped cross-validation it would let the
    model be tested on a recording it was trained on.
    """
    import hashlib

    if table.empty:
        return table
    digests = table["path"].map(lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest())
    duplicated = digests.duplicated()
    if duplicated.any():
        names = ", ".join(Path(p).name for p in table.loc[duplicated, "path"])
        warn(f"Skipped {duplicated.sum()} duplicate {kind} file(s), identical to "
             f"another file: {names}. Record a new take instead of copying.")
    return table.loc[~duplicated].reset_index(drop=True)


# ----------------------------------------------------------------- images ---

def load_image(path: Path | str) -> np.ndarray:
    """Read any image file as an RGB uint8 array, applying EXIF rotation."""
    path = Path(path)
    try:
        with Image.open(path) as img:
            img.seek(0)  # first frame of a GIF / multi-page TIFF
            return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    except Exception as pil_error:  # noqa: BLE001 - try OpenCV before giving up
        import cv2

        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII paths
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if bgr is None:
            hint = " Install pillow-heif for HEIC photos." if path.suffix.lower() in {
                ".heic", ".heif", ".avif"} and pillow_heif is None else ""
            raise ValueError(f"Cannot read image {path}: {pil_error}.{hint}") from pil_error
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ audio ---

def _decode_with_pyav(path: Path, sr: int) -> np.ndarray:
    """Decode any audio container FFmpeg understands to mono float32 at ``sr``."""
    import av

    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"{path} contains no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    if not chunks:
        raise ValueError(f"{path} decoded to an empty signal")
    return np.concatenate(chunks).astype(np.float32)


def load_audio(path: Path | str, sr: int = 16_000) -> np.ndarray:
    """Read any audio file as mono float32 resampled to ``sr`` Hz.

    soundfile is tried first (fast, exact for WAV). If it cannot read the file,
    for example an M4A recording, or one renamed to ``.wav``, PyAV decodes it.
    """
    import soundfile as sf

    path = Path(path)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception as sf_error:  # noqa: BLE001 - fall back to FFmpeg via PyAV
        try:
            return _decode_with_pyav(path, sr)
        except ImportError as missing:
            raise ImportError(
                f"{path.name} is not a format soundfile can read. Install PyAV "
                "(`pip install av`) to read M4A, AAC, MP3, WEBM and other phone formats."
            ) from missing
        except Exception as av_error:  # noqa: BLE001
            raise ValueError(f"Cannot read audio {path}: {sf_error}; {av_error}") from av_error
    if file_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def detect_format(path: Path | str) -> str:
    """Describe the real format of a file from its first bytes (ignores the extension)."""
    head = Path(path).read_bytes()[:16]
    if head.startswith(b"RIFF"):  # RIFF holds WAV audio, WEBP images and AVI video
        return {b"WAVE": "wav", b"WEBP": "webp", b"AVI ": "avi"}.get(head[8:12], "riff")
    signatures = [
        (b"fLaC", "flac"), (b"OggS", "ogg"), (b"ID3", "mp3"),
        (b"\xff\xd8\xff", "jpeg"), (b"\x89PNG", "png"), (b"GIF8", "gif"),
        (b"BM", "bmp"), (b"#!AMR", "amr"), (b"\x1aE\xdf\xa3", "webm"),
    ]
    for magic, name in signatures:
        if head.startswith(magic):
            return name
    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("latin-1").strip().lower()
        if brand in {"heic", "heix", "mif1", "msf1", "heim", "heis"}:
            return "heic"
        if brand.startswith("avif"):
            return "avif"
        return "mp4/m4a"
    if head[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    if head[:2] in {b"\xff\xf1", b"\xff\xf9"}:
        return "aac"
    return "unknown"


# ------------------------------------------------------------ conversion ---

def normalise_data(data_dir: Path = REPO_ROOT / "data", dry_run: bool = False,
                   wav_rate: int = 44_100) -> list[str]:
    """Convert every member file to canonical format and name, in place.

    * audio -> ``data/audio/<member>/<member>_<phrase>_<take>.wav`` (16-bit mono)
    * images -> ``data/images/<member>/<member>_<expression>.jpg``

    Files already in the right format and name are left untouched. The original
    of a converted file is deleted only after the new file is written.
    """
    import soundfile as sf

    actions = []
    for member_dir in sorted(p for p in (data_dir / "audio").glob("*") if p.is_dir()):
        member = slug(member_dir.name)
        for path in sorted(p for p in member_dir.iterdir() if is_media_file(p, "audio")):
            phrase, take = phrase_and_take(path.stem, member)
            target = data_dir / "audio" / member / f"{member}_{phrase}_{take}.wav"
            real = detect_format(path)
            if path == target and real == "wav":
                continue
            while target.exists() and not target.samefile(path):  # never overwrite another take
                take += 1
                target = target.with_name(f"{member}_{phrase}_{take}.wav")
            actions.append(f"audio: {path.relative_to(data_dir)} ({real}) -> {target.relative_to(data_dir)}")
            if not dry_run:
                y = load_audio(path, wav_rate)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp.wav")
                sf.write(tmp, y, wav_rate, subtype="PCM_16")
                if path != target:
                    path.unlink()
                tmp.replace(target)
    for member_dir in sorted(p for p in (data_dir / "images").glob("*") if p.is_dir()):
        member = slug(member_dir.name)
        for path in sorted(p for p in member_dir.iterdir() if is_media_file(p, "image")):
            expression = expression_from_name(path.stem, member)
            real = detect_format(path)
            keep_suffix = path.suffix.lower() if real in {"jpeg", "png"} else ".jpg"
            target = data_dir / "images" / member / f"{member}_{expression}{keep_suffix}"
            if path == target:
                continue
            n = 1
            while target.exists() and not target.samefile(path):  # never overwrite another photo
                n += 1
                target = target.with_name(f"{member}_{expression}_{n}{keep_suffix}")
            actions.append(f"image: {path.relative_to(data_dir)} ({real}) -> {target.relative_to(data_dir)}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                if real in {"jpeg", "png"}:
                    path.rename(target)
                else:
                    Image.fromarray(load_image(path)).save(target, quality=95)
                    path.unlink()
    report_loose_files(data_dir / "images", "image")
    report_loose_files(data_dir / "audio", "audio")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert data/ files to canonical formats and names.")
    parser.add_argument("--dry-run", action="store_true", help="only list what would change")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()
    actions = normalise_data(args.data_dir, dry_run=args.dry_run)
    print("\n".join(actions) if actions else "Nothing to convert: every file is already canonical.")


if __name__ == "__main__":
    main()
