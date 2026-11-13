# Multimodal Data Preprocessing (Formative 2)

A user identity and product recommendation system. A user is recognised by their **face**, a product recommendation is
computed for their customer account, and it is only released after their **voice** confirms the transaction.

```
face photo --> face recognition --(unknown)--> ACCESS DENIED
                 | member M
                 v
         product model predicts for M (held back)
                 |
voice clip --> voiceprint verification (is it M?) --(no)--> ACCESS DENIED
                 v
         recommendation displayed
```

## Quick start

Works the same on Windows, macOS, Linux, Google Colab and Kaggle (Python 3.9 or newer).

```bash
git clone https://github.com/mutaganzwa-desire/multimodal_data_preprocessing.git
cd multimodal_data_preprocessing
pip install -r requirements.txt
python app.py --demo
```

No system programs (such as FFmpeg) are needed: every dependency installs with `pip`. The notebooks and the app also
install any missing package themselves, so opening a notebook directly also works. On Colab or Kaggle, open a notebook
from GitHub and run it: the first cell clones the repository.

## The command-line app

```bash
python app.py --demo                            # authorised transaction + unauthorised attempts
python app.py                                   # interactive: asks for a photo, then a voice clip
python app.py --face PHOTO --voice CLIP         # one transaction (exit code 0 = granted, 1 = denied)
python app.py --face PHOTO --voice CLIP --customer-id 101 --date 2026-12-01
python app.py --retrain                         # rebuild the three models first
```

Each member is linked to a customer account in `data/member_accounts.csv`. To test with a real stranger, put their photo
and voice recording in `data/unauthorized/` (see the README there).

## Notebooks

Run in order (each one also works on its own, because the generated files are committed):

| # | Notebook | Produces |
|---|---|---|
| 1 | `Formative_2_Tabular_Data_Preprocessing.ipynb` | EDA, cleaning, merge, `datasets/merged_customer_recommendation_dataset.csv` |
| 2 | `Formative_2_Image_Processing.ipynb` | face crops, augmentations, `datasets/image_features.csv` |
| 3 | `Formative_2_Audio_Processing.ipynb` | waveforms, spectrograms, augmentations, `datasets/audio_features.csv` |
| 4 | `Formative_2_Models_and_System_Demo.ipynb` | the three models, their metrics, and the system simulation |

## Repository layout

```
app.py                      command-line app (the system simulation)
src/
  media_io.py               reads any image/audio format; `python -m src.media_io` tidies data/
  image_pipeline.py         face detection, cropping, augmentation, image features
  audio_pipeline.py         speech trimming, augmentation, audio features
  verification.py           shared face/voice training, evaluation and accept/reject logic
  face_model.py             facial recognition model        (python -m src.face_model)
  voice_model.py            voiceprint verification model   (python -m src.voice_model)
  product_model.py          product recommendation model    (python -m src.product_model)
  model_store.py            saves models; retrains them automatically on another scikit-learn version
  environment.py            installs missing packages
data/images/<member>/       3 photos per member: neutral, smiling, surprised
data/audio/<member>/        "yes approve" and "confirm transaction" recordings
data/unauthorized/          optional stranger samples for the demo (never used for training)
datasets/                   raw CSVs, merged dataset, image_features.csv, audio_features.csv
models/                     trained models (.joblib)
notebooks/                  the four notebooks above
reports/                    project report
```

## Adding or replacing a member's samples

1. Put the files in `data/images/<member>/` and `data/audio/<member>/`, for example `arnold_smiling.heic` or
   `arnold_yes_approve_2.m4a`. Any common format and naming variant works (`Smile`, `Yes Approve 2`...).
2. Optionally run `python -m src.media_io` to convert everything to standard JPEG/WAV with standard names
   (`--dry-run` shows what would change).
3. Rebuild the features and models:

   ```bash
   python -m src.image_pipeline
   python -m src.audio_pipeline
   python -m src.face_model
   python -m src.voice_model
   ```

   or rerun notebooks 2, 3 and 4.

Supported formats: images in JPEG, PNG, HEIC/HEIF, WEBP, AVIF, BMP, TIFF and GIF; audio in WAV, M4A, AAC, MP3, MP4, OGG,
OPUS, FLAC, WEBM, AMR, 3GP, AIFF and WMA. Files are decoded by content, so a wrong extension does not matter. Photos
outside a member folder and byte-identical duplicate files are reported with a warning.
