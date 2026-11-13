"""User identity and product recommendation system (command-line app).

Flow, as in the assignment diagram:

    face photo --> face recognition --(unknown)--> ACCESS DENIED
        | recognised member
        v
    product recommendation model runs for the member's customer account
        |
    voice recording --> voiceprint verification against the SAME member
        |                     --(mismatch / unknown)--> ACCESS DENIED
        v
    prediction approved and displayed

Usage (from the repository root, any OS):

    python app.py                                   # interactive: asks for the files
    python app.py --face PHOTO --voice CLIP         # one transaction
    python app.py --face PHOTO --voice CLIP --customer-id 101 --date 2026-11-20
    python app.py --demo                            # scripted authorised + unauthorised attempts
    python app.py --retrain                         # rebuild the three models first

Photos and recordings can be in any common format (JPEG, PNG, HEIC, WEBP /
WAV, M4A, MP3, AAC, OGG, OPUS...).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.environment import ensure_packages  # noqa: E402

LINE = "-" * 68


def _say(text: str = "") -> None:
    print(text, flush=True)


def _status(ok: bool, text: str) -> None:
    _say(f"  [{'PASS' if ok else 'FAIL'}] {text}")


class RecommendationSystem:
    """Loads the three models once and runs transactions."""

    def __init__(self, retrain: bool = False):
        import pandas as pd

        from src import face_model, product_model, voice_model
        from src.model_store import save

        if retrain:
            for module in (face_model, voice_model, product_model):
                save(module.train(verbose=False), module.MODEL_PATH)
        self.face = face_model.load()
        self.voice = voice_model.load()
        self.product = product_model.load()
        accounts = REPO_ROOT / "data" / "member_accounts.csv"
        self.accounts = (pd.read_csv(accounts).set_index("member")["customer_id"].to_dict()
                         if accounts.exists() else {})

    # ------------------------------------------------------------- steps ---
    def check_face(self, photo: Path, model=None):
        from src import face_model

        _say(f"\nStep 1 - Face recognition: {photo.name}")
        decision = face_model.recognise(model or self.face, photo)
        _status(decision.accepted, decision.reason)
        return decision

    def run_prediction(self, member: str, customer_id: int | None, visit_date: dt.date | None):
        _say("\nStep 2 - Product recommendation model")
        customer_id = customer_id if customer_id is not None else self.accounts.get(member)
        if customer_id is None:
            _status(False, f"no customer account is linked to {member} (see data/member_accounts.csv)")
            return None, None
        try:
            ranking = self.product.recommend(int(customer_id), visit_date)
        except KeyError as error:
            _status(False, str(error).strip("'\""))
            return None, None
        _status(True, f"prediction computed for customer {customer_id}; held until the voice "
                      "check approves it")
        return customer_id, ranking

    def check_voice(self, clip: Path, member: str, model=None):
        from src import voice_model

        _say(f"\nStep 3 - Voiceprint verification for {member}: {clip.name}")
        decision = voice_model.verify(model or self.voice, clip, member)
        if decision.accepted:
            _status(True, f"voice approved: {decision.reason}")
        else:
            heard = max(decision.probabilities, key=decision.probabilities.get) if decision.probabilities else None
            extra = f" (the voice sounds most like {heard})" if heard and heard != member else ""
            _status(False, decision.reason + extra)
        return decision

    # ------------------------------------------------------- transaction ---
    def transaction(self, photo: Path, clip: Path | None, customer_id: int | None = None,
                    visit_date: dt.date | None = None, face_model=None, voice_model=None,
                    ask_for_voice: bool = False) -> bool:
        """Run the full flow. Returns True when the prediction is shown."""
        face = self.check_face(photo, face_model)
        if not face.accepted:
            return _denied("face not recognised")
        _say(f"  Welcome, {face.member}.")

        customer_id, ranking = self.run_prediction(face.member, customer_id, visit_date)
        if ranking is None:
            return _denied("no customer data for the prediction")

        if clip is None and ask_for_voice:
            clip = _ask_path("Path to the voice recording that confirms the transaction: ")
        if clip is None:
            return _denied("no voice confirmation given")
        voice = self.check_voice(clip, face.member, voice_model)
        if not voice.accepted:
            return _denied("voice not verified")

        _say(f"\n{LINE}\n  ACCESS GRANTED - recommendation for {face.member} "
             f"(customer {customer_id})\n{LINE}")
        _say(f"  Recommended product category: {ranking.index[0]}  ({ranking.iloc[0]:.0%})")
        for category, p in ranking.iloc[1:].items():
            _say(f"    then {category:<12} {p:.0%}")
        return True


def _denied(why: str) -> bool:
    _say(f"\n{LINE}\n  ACCESS DENIED - {why}\n{LINE}")
    return False


def _ask_path(prompt: str) -> Path | None:
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        _say(f"  File not found: {path}")


# ------------------------------------------------------------------- demo ---

def _first(folder: Path, pattern: str) -> Path | None:
    from src import media_io

    kind = "image" if "images" in folder.parts else "audio"
    files = sorted(p for p in folder.glob(pattern) if media_io.is_media_file(p, kind)) if folder.exists() else []
    return files[0] if files else None


def run_demo(system: RecommendationSystem) -> None:
    """Scripted scenarios for the system simulation video."""
    from src import face_model, voice_model
    from src import media_io

    images, audio = REPO_ROOT / "data" / "images", REPO_ROOT / "data" / "audio"
    voiced = [m for m in system.voice.classes if m in system.face.classes]
    results = []

    def scenario(title, **kwargs):
        _say(f"\n\n{'=' * 68}\n  SCENARIO {len(results) + 1}: {title}\n{'=' * 68}")
        results.append((title, system.transaction(**kwargs)))

    member = voiced[0]
    scenario(f"authorised transaction by {member}",
             photo=_first(images / member, f"{member}_neutral.*"),
             clip=_first(audio / member, f"{member}_yes_approve_*"))

    impostor = voiced[1]
    scenario(f"{impostor} uses {member}'s face but confirms with their own voice",
             photo=_first(images / member, f"{member}_smiling.*"),
             clip=_first(audio / impostor, f"{impostor}_confirm_transaction_*"))

    strangers = REPO_ROOT / "data" / "unauthorized"
    stranger_photos = sorted(p for p in strangers.glob("*") if media_io.is_media_file(p, "image")) \
        if strangers.exists() else []
    stranger_clips = sorted(p for p in strangers.glob("*") if media_io.is_media_file(p, "audio")) \
        if strangers.exists() else []

    if stranger_photos:
        scenario("a stranger (not in the team) tries to log in",
                 photo=stranger_photos[0], clip=stranger_clips[0] if stranger_clips else None)
    else:
        outsider = system.face.classes[-1]
        _say(f"\n(No photo in data/unauthorized/. Simulating a stranger: the face model is "
             f"re-enrolled without {outsider}, so {outsider} is someone the system has never seen.)")
        scenario(f"a stranger ({outsider}, not enrolled) tries to log in",
                 photo=_first(images / outsider, f"{outsider}_smiling.*"), clip=None,
                 face_model=face_model.train(exclude=(outsider,), verbose=False, reuse=system.face))

    if stranger_clips:
        scenario(f"{member}'s face, but a stranger's voice approves",
                 photo=_first(images / member, f"{member}_surprised.*"), clip=stranger_clips[0])
    elif len(voiced) >= 3:
        outsider = voiced[-1]
        _say(f"\n(No recording in data/unauthorized/. Simulating a stranger's voice: the voice "
             f"model is re-enrolled without {outsider}.)")
        scenario(f"{member}'s face, but a stranger's voice ({outsider}, not enrolled) approves",
                 photo=_first(images / member, f"{member}_surprised.*"),
                 clip=_first(audio / outsider, f"{outsider}_yes_approve_*"),
                 voice_model=voice_model.train(exclude=(outsider,), verbose=False, reuse=system.voice))

    _say(f"\n\n{'=' * 68}\n  SUMMARY\n{'=' * 68}")
    for i, (title, granted) in enumerate(results, 1):
        _say(f"  {i}. {'GRANTED' if granted else 'DENIED '}  {title}")


# ------------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="User identity and product recommendation system.")
    parser.add_argument("--face", type=Path, help="photo of the user (any image format)")
    parser.add_argument("--voice", type=Path, help="recording that confirms the transaction (any audio format)")
    parser.add_argument("--customer-id", type=int, help="customer account to predict for "
                        "(default: the account linked to the recognised member)")
    parser.add_argument("--date", type=dt.date.fromisoformat, help="visit date, YYYY-MM-DD (default: today)")
    parser.add_argument("--demo", action="store_true", help="run the scripted authorised/unauthorised scenarios")
    parser.add_argument("--retrain", action="store_true", help="retrain all three models before running")
    args = parser.parse_args()

    ensure_packages()
    _say(f"{LINE}\n  User identity and product recommendation system\n{LINE}")
    system = RecommendationSystem(retrain=args.retrain)
    _say(f"  Enrolled faces : {', '.join(system.face.classes)}")
    _say(f"  Enrolled voices: {', '.join(system.voice.classes)}")

    if args.demo:
        run_demo(system)
        return
    if args.face:
        for path in filter(None, (args.face, args.voice)):
            if not path.is_file():
                parser.error(f"file not found: {path}")
        granted = system.transaction(args.face, args.voice, args.customer_id, args.date)
        sys.exit(0 if granted else 1)

    _say("\nInteractive mode. Press Enter on an empty line to quit.")
    while True:
        photo = _ask_path("\nPath to the face photo: ")
        if photo is None:
            break
        system.transaction(photo, None, args.customer_id, args.date, ask_for_voice=True)


if __name__ == "__main__":
    main()
