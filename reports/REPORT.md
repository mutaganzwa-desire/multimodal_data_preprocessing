# Formative 2: Multimodal Data Preprocessing - Report

**System:** user identity and product recommendation with face recognition and voiceprint verification
**Repository:** https://github.com/mutaganzwa-desire/multimodal_data_preprocessing
**Simulation video:** _add link_
**Team:** Alain, Arnold, Desire, Ivan

> Status: Ivan has one recording per phrase (his `_2` files were copies of his `_1` files and were deleted). If he
> records second takes, rerun notebooks 3 and 4 (or `python -m src.audio_pipeline` then `python -m src.voice_model`) and update the
> voice figures in sections 4, 6 and 7.

---

## 1. System overview

The system decides whether a person may see a personalised product recommendation:

1. **Face recognition.** A photo is matched against the enrolled team members. An unknown face is denied immediately.
2. **Product prediction.** For the recognised member, the product recommendation model predicts the category their
   customer account is most likely to buy. The result is held back.
3. **Voice verification.** The member says "Yes, approve" or "Confirm transaction". The voice must belong to *the same
   member* the face check recognised. Only then is the prediction displayed; otherwise access is denied.

Everything is implemented as reusable Python modules in `src/`, a command-line app (`app.py`) and four notebooks that
call the same functions, so the notebook numbers are the app's numbers.

## 2. Tabular data: EDA, cleaning and merge (notebook 1)

**Data.** `customer_social_profiles` has 155 rows (84 customers, IDs such as `A178`, 5 platforms, engagement score
50-99, purchase interest 1.0-5.0, review sentiment). `customer_transactions` has 150 rows (one per transaction, 75
customers with integer IDs, one purchase per day from 2024-01-01 to 2024-05-29, 5 product categories).

**EDA.** Four labelled plots: distributions of engagement and purchase interest (neither strongly skewed: skewness
-0.11 and -0.23), purchase amount by category with an IQR outlier check (no outliers, bounds -135 to 703), a correlation
heatmap (engagement and purchase interest almost uncorrelated, r = 0.09), and the target balance (25 to 35 transactions
per category, balanced enough that accuracy is meaningful and no resampling is needed).

**Cleaning.**
- 5 exact duplicate social records removed.
- IDs harmonised: `A178` and `178` are the same customer, so both tables get an integer `customer_id` (range 100-199).
- `purchase_date` converted from text to a date.
- 10 missing ratings (6.7%) filled with the median (3.0), with a `customer_rating_missing` flag kept.

**Join logic.** A customer can have several social records but each transaction must stay one row. Social records are
therefore first aggregated to one row per customer (record count, platform count, mean and max engagement, mean and
standard deviation of purchase interest, their product, sentiment shares, dominant platform), then **left-joined** onto
the transactions with `validate="many_to_one"`.

**Post-merge checks.** 150 rows before and after the merge, `transaction_id` still unique, no missing values,
sentiment shares sum to 1. 117 transactions matched a profile; the 33 transactions from 14 customers without a profile
are kept (dropping them would lose 22% of the data), with `has_social_profile = 0` and the profiled customers' typical
values filled in. Date features (`purchase_month`, `purchase_dayofweek`) are added. The purchase amount and rating are
kept in the file but excluded from the model inputs, because they only exist after the purchase.

**Output:** `datasets/merged_customer_recommendation_dataset.csv` (150 rows x 21 columns).

## 3. Image data (notebook 2)

**Collection.** Every member submitted a neutral, a smiling and a surprised photo: 12 photos, named
`<member>_<expression>.<ext>` in `data/images/<member>/`. Resolutions range from 384 x 432 to 2448 x 3264.

**Face detection and cropping.** Backgrounds differ between members, so features are taken from the face only. OpenCV's
Haar detector runs on the photo and on copies rotated by up to 30 degrees, and a candidate counts only if eyes are found
in its upper part. This found a face in all 12 photos, including Desire's smiling photo, where the head is tilted 30
degrees. The face is cropped with a 15% margin and resized to 128 x 128.

**Augmentation.** 7 augmentations per photo, each imitating a login condition: rotation of +15 and -15 degrees (tilted
head), horizontal flip (mirrored selfie camera), grayscale, brighter and darker (lighting), Gaussian noise (low-light
sensor). 12 photos x 8 versions = 96 rows.

**Features.** 532 per face: a 48-value RGB colour histogram (skin tone, lighting), 160 uniform LBP texture values over a
4 x 4 grid, and 324 HOG shape values (eyes, nose, mouth, jaw). Saved to `datasets/image_features.csv` with metadata
columns (member, expression, augmentation, source image).

**Sanity check.** A leave-one-photo-out nearest-neighbour test matches 0.92 of held-out versions to the right member
(0.25 by chance).

## 4. Audio data (notebook 3)

**Collection.** Each member records "Yes, approve" and "Confirm transaction", ideally twice. Alain, Arnold and Desire
have 4 recordings each; Ivan has 2, because his second takes were byte-identical copies of the first and were removed.

**Cleaning.** Members used different phones, with different silence and loudness, so a model could otherwise tell
speakers apart by clip length or volume. Each clip is converted to mono 16 kHz, cut to the spoken phrase, and
peak-normalised. A frame counts as speech when it is within 30 dB of the peak and at least 12 dB above the recording's
own noise floor; the second rule was added for Arnold's first "confirm transaction", where a steady hum only 20 dB
below his voice made a fixed rule keep the whole 3.1 s file. Sound regions are joined across gaps shorter than 0.4 s
and the most energetic one is kept, which drops stray clicks.

**Visualisation.** Raw waveforms with the kept region shaded, cleaned waveforms, and mel spectrograms for every clip,
each interpreted in the notebook (syllable bursts, speaking speed, harmonics, and a device difference in the noise
floor between phones).

**Augmentation.** 6 per recording: pitch +2 and -2 semitones, time stretch 15% faster and slower, background noise at
20 dB SNR, and 6 dB quieter.

**Features.** 50 per clip, frame statistics over time: 13 MFCC means and standard deviations, 13 MFCC delta standard
deviations, spectral roll-off, centroid and bandwidth, RMS energy, zero-crossing rate, and duration. Saved to
`datasets/audio_features.csv` (98 rows). A leave-one-recording-out nearest-neighbour check matches 0.58 of held-out
versions to the right speaker (0.25 by chance). Arnold and Ivan sit close together in the feature space, and Ivan's
single recording per phrase makes him the hardest speaker to hold out.

## 5. Handling any file format, on any machine

Members' phones produced different formats, and some files had the wrong extension: Ivan's `.wav` files were really
MP4/AAC recordings. `src/media_io.py` reads files by their content: images through Pillow (with `pillow-heif` for
iPhone HEIC) and audio through `soundfile`, falling back to PyAV, which bundles FFmpeg inside its pip package. JPEG,
PNG, HEIC, WEBP, AVIF and BMP photos, and WAV, M4A, AAC, MP3, OGG, OPUS, FLAC and WEBM recordings, all work with no
system install. `python -m src.media_io` converts the data folder to standard WAV/JPEG with standard names. Photos left
outside a member folder and duplicate files are reported instead of silently used.

Each notebook's first cell finds the repository from any folder, clones it when opened alone (Colab, Kaggle), and
installs only missing packages. Saved models record their scikit-learn version and retrain automatically under
another version. All four notebooks and the app were verified from a fresh clone in an empty Python environment.

## 6. Models and evaluation (notebook 4)

### Face and voice: identification plus verification

For both biometric models, Logistic Regression, Random Forest and XGBoost are compared with **leave-one-group-out
cross-validation**, where a group is one source photo or recording. This keeps every augmented copy of a sample on the
same side of the split; a random row split would test on near-copies of the training data and inflate the scores. The
model with the lowest log loss is kept, because the decision uses its probabilities.

A classifier always names one of the members it knows, even for a stranger, so a sample is accepted only if **both**:
- the classifier's probability for the claimed member is at least a probability threshold, and
- the sample's mean cosine similarity to that member's 3 closest enrolled feature vectors is at least a similarity
  threshold.

Both thresholds are tuned on held-out attempts of three kinds: genuine members, teammates claiming to be another
member, and strangers (each member removed from training in turn). The thresholds maximise the average of the genuine
acceptance rate and the two rejection rates.

| Model | Accuracy | Macro F1 | Log loss | Random Forest (acc / F1 / loss) | XGBoost (acc / F1 / loss) |
|---|---|---|---|---|---|
| Face: Logistic Regression | **0.917** | **0.914** | **0.177** | 0.906 / 0.901 / 0.636 | 0.802 / 0.791 / 0.614 |
| Voice: Logistic Regression | **0.724** | **0.629** | **1.002** | 0.582 / 0.482 / 1.155 | 0.622 / 0.536 / 1.424 |

| Check | Thresholds (probability / similarity) | Genuine accepted | Teammates rejected | Strangers rejected |
|---|---|---|---|---|
| Face | 0.80 / 0.36 | 91% (92% of unmodified photos) | 100% | 99% |
| Voice | 0.50 / 0.40 | 66% (79% of unmodified recordings) | 94% | 80% |

The only face errors are the 8 versions of Desire's neutral photo, the one photo taken in a different place and light
from her other two. The similarity check is essential: strangers often get a classifier probability of 0.4 to 0.9 for
some member, but rarely look similar to that member's enrolled samples.

The voice check is the weaker one. Alain is always recognised and Desire in 25 of 28 versions, but Arnold is confused
with Ivan in 9 of 28 versions, and Ivan is never recognised when one of his recordings is held out (all 14 versions are
taken for Arnold). With only one usable recording per phrase, holding one out leaves a single example of Ivan's voice,
saying the other phrase. Trained on all recordings, as in the app, the model accepts all of Ivan's own recordings
(probability about 0.7), but the held-out test shows his voice is not yet reliably separated from Arnold's. The cause is
the amount of audio, not the method: second takes from Ivan and a few more takes per member are the most
effective improvement.

### Product recommendation

Inputs are the customer's aggregated social profile and the visit date (purchase amount and rating excluded, as they
occur after the purchase). Evaluation is 5-fold cross-validation repeated 5 times, **grouped by customer**, against a
most-frequent-class baseline. The model with the best macro F1 is kept.

| Model | Accuracy | Macro F1 | Log loss |
|---|---|---|---|
| Baseline (most frequent) | 0.224 | 0.111 | 1.602 |
| **Logistic Regression (selected)** | **0.259** | **0.230** | 1.672 |
| Random Forest | 0.216 | 0.175 | 1.722 |
| XGBoost | 0.208 | 0.190 | 1.844 |

The features describe how engaged a customer is, not what they like, and the EDA showed nearly identical purchase
interest across categories. With 150 transactions from 75 customers, the category is close to unpredictable: Logistic
Regression beats the baseline on accuracy and clearly on macro F1, but every model's log loss is above the baseline's.
The app therefore shows the full ranked list of categories with their probabilities rather than a single confident
answer. Richer behavioural data (product views, clicks) would be the main way to improve it.

## 7. Multimodal logic and system simulation

The modalities guard different steps: the face decides who may run a prediction, and the voice decides whether its
result is released. The voice is verified against the identity established by the face, not against "any member".
`python app.py --demo` (also run in notebook 4) shows:

| # | Scenario | Result | Why |
|---|---|---|---|
| 1 | Alain: own face, own voice | **Granted** | face 1.00 / 0.93, voice 0.96 / 0.94; recommendation shown |
| 2 | Arnold holds Alain's photo, confirms with his own voice | **Denied at step 3** | voice probability for Alain 0.014 |
| 3 | Stranger's face (system enrolled without Ivan) | **Denied at step 1** | similarity to the nearest member 0.17 < 0.36 |
| 4 | Alain's face, stranger's voice (enrolled without Ivan) | **Denied at step 3** | voice probability for Alain 0.11 < 0.50 |

When every member in turn plays the stranger, all 12 stranger photos are rejected, and 34 of 42 stranger voice
attempts (81%), in line with the held-out rates above.
A real outsider's photo and recording can be placed in `data/unauthorized/`, and the demo then uses them. The app also
runs interactively (`python app.py`) or for one transaction
(`python app.py --face photo.heic --voice clip.m4a`), and denies photos where no face is detected.

## 8. Limitations and next steps

- **Small biometric dataset:** one photo per expression and 2 to 4 recordings per member. More photos in varied places
  and light, and more takes, would help most. The face model's one error and the Arnold/Ivan voice confusion both come
  from this.
- **Device bias:** each member used a different phone, so part of the voice signal may be the device. The stranger
  recording should be made on a team phone.
- **Weak product signal:** the tabular features barely predict the category; the model is a ranked suggestion.
- **Stranger thresholds** are estimated by leaving one member out, since no real outsiders were collected yet.

## 9. Team contributions

| Member | Contributions (from the repository history; to be confirmed by the team) |
|---|---|
| Desire | Project setup and raw datasets; tabular EDA, cleaning notes, merge validation and processed dataset; own photos and recordings |
| Alain | Image and audio preprocessing pipelines and feature files; image and audio notebooks; tabular notebook portability and cleaning; README; own photos and recordings |
| Ivan | Own photos and recordings |
| Arnold | Own photos and recordings |
| _to fill_ | Any-format media loading, notebook portability, face/voice/product models, command-line app, models notebook, report |

See also the contribution sheet.
