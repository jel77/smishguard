# 🛡️ SmishGuard

**Detecting banking and online-payment smishing in Filipino/Taglish SMS** using Logistic
Regression, Decision Tree and Support Vector Machine (SVM) classifiers.

SmishGuard is a Streamlit web app that labels an SMS as **legit** or **smishing** (SMS phishing)
from the message text and sender. It trains and compares three models, tests them statistically,
and shows which features give scams away. It was built for the study described in
[Chapter 1 & 2_Team TRIOjan_Virus.pdf](Chapter%201%20%26%202_Team%20TRIOjan_Virus.pdf).

**Team TRIOjan_Virus (BCS33):** Cruz, Hannah Ellise L. · Rios, Jakob Theodore Y. · Tagum, Jel Mark P.
De La Salle University – Dasmariñas, College of Information and Computer Studies.
Software Engineering 1 Lecture, submitted to Ms. Sheryl Kamantigue.

---

## Quick start

Requires **Python 3.12+** (numpy and scipy need it; tested on 3.14). Windows commands:

```powershell
git clone <this repo>
cd smishguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

One dataset can't be shared in this repo: download the ASEA file into `datasets/` as described in
[datasets/README.md](datasets/README.md#getting-the-files-and-licenses) before running, or your
scores won't match the ones below.

The app opens at <http://localhost:8501>. On macOS/Linux, activate with `source .venv/bin/activate`.

To run the tests:

```powershell
pip install -r requirements-dev.txt
pytest                 # everything, including a ~15s run of the whole app
pytest -m "not slow"   # skip the app run
```

---

## How it works

```
SMS text + sender
   │
   ├─ clean(): lowercase, links → "URL <words in link>", digits → "NUM"
   │     └─ TF-IDF of words and 2-word phrases
   │
   └─ hand(): 11 hand-made features (below), scaled to 0..1
         │
         ▼
  Logistic Regression · Decision Tree · linear SVM   (+ random-guess baseline)
         │
         ▼
  smishing probability → SMISHING if ≥ cutoff (default 0.50)
```

**Sender dropout.** Every training message that has a sender is also added once without it.
Most sender-less training messages are legit (the English Mendeley set has no senders), so without
this the models learn "no sender = legit" and miss scams in `sms_taglish.csv`, which has no senders.
In a 5-fold × 3 check, it raised SVM recall on those messages from 0.72 to 0.93 and LR's from
0.92 to 0.96, with overall F1 unchanged (0.98).

The SVM is scikit-learn's `LinearSVC`, calibrated to give probabilities. It's the same kind of
model as `SVC(kernel="linear")` but about 50× faster on thousands of messages.

| Feature | What it measures |
|---|---|
| `has_url` | Message contains a link (`http…`, `www.…`, or a bare domain like `gcash-verify.top`) |
| `short_url` | Link uses a shortener (`bit.ly`, `cutt.ly`, `tinyurl.com`, `t.me`, …) |
| `suspicious_tld` | Link ends in a cheap domain often used by scammers (`.xyz`, `.top`, `.cc`, `.online`, `.info`, …) |
| `urgency_count` | Urgency words: *kailangan agad*, *agad*, *ngayon*, *claim now*, *urgent*, *expire*, … |
| `money_count` | Money, prize and account words: *php*, *₱*, *premyo*, *panalo*, *gcash*, *maya*, *bdo*, … |
| `length` | Number of characters |
| `special_chars` | Number of symbols/punctuation |
| `has_phone` | Message contains a PH mobile number (`09xx xxx xxxx`, `+639…`) |
| `sender_is_number` | Sender is a plain mobile number |
| `sender_is_alpha` | Sender is a named ID or short code (`BDO`, `GCash`, `2958`) |
| `sender_unknown` | No sender given |

The feature code is in [features.py](features.py).

---

## Using the app

| Tab | What it does |
|---|---|
| 🔎 **Live demo** | Paste a message (and optionally its sender). Each model gives a smishing probability, and a vote shows how many models flag it. You can flag wrong results. |
| 🚩 **Feedback** | The messages you flagged. Retrain the models with them, download them as `flagged.csv`, or go back to the original model. |
| 📊 **Model comparison** | Held-out test results (overall and per source file), a recall vs false-alarm chart across cutoffs, **cross-validation** with **McNemar's significance test**, and a **recommendation** for which model to deploy. |
| 🧠 **Top features** | Hand-made features ranked across all 3 models, permutation importance from cross-validation, a **feature-set comparison** (words only vs hand-made only vs both, with McNemar's test), and each model's top features (including the Decision Tree's rules). |
| 🗂️ **Dataset** | The core dataset, plus a per-file report for anything in `datasets/`. |

The **Smishing cutoff** slider in the sidebar sets the probability at which a message counts as
smishing. Lower it to catch more scams (higher recall) at the cost of more false alarms.

> Flagged messages are kept in memory, are shared by everyone using the same running app, and are
> lost when it restarts. To keep them, download `flagged.csv` and add the rows to `sms_taglish.csv`.

---

## Data

### Core Filipino/Taglish set

All train/test splits and cross-validation use the **core set**, so every score measures
Filipino/Taglish smishing. It is `sms_taglish.csv` plus the `datasets/` files with role `core`,
deduplicated by template (the same message with a different OTP, amount or link counts once):

| Source | Smishing | Legit | Senders? |
|---|---|---|---|
| `sms_taglish.csv` (our own) | 104 | 95 | No |
| ASEA (GitHub) | 2,311 | 148 | Yes (masked numbers, sender IDs) |
| Tagalog-SMS (Kaggle, `tagalog-sms.xlsx`) | 167 | 1,218 | Yes |
| **Total** | **2,582** | **1,461** | |

Smishing covers banking, e-wallet and other money/account lures (prize, raffle, loan,
investment, pasaload, delivery, government spoofing) plus gambling scams. Marketing, job scams
and unlabelled spam are left out. The Dataset tab shows the current counts.

`sms_taglish.csv` columns:

| Column | Values |
|---|---|
| `sender` | Optional. A mobile number (`09171234567`), a sender ID (`BDO`, `GCash`) or blank. Blank for all current rows. |
| `text` | The SMS message |
| `label` | `1` = smishing, `0` = legit |

### Training-only data and other files: `datasets/`

Drop CSV or Excel files (e.g. from Kaggle) into [datasets/](datasets/) and refresh the app. Each
file is either `core` (joins the split above) or `train` (training only, e.g. the English
Mendeley smishing set). Columns are detected automatically, and labels are mapped to 1/0 per
file in `datasets/config.json`. See [datasets/README.md](datasets/README.md) for the files in
use, how each one's labels are mapped, and their sources and licenses.

**Data sources:** Tagalog-SMS by onzero0 (Kaggle, CC0); SMS Phishing Dataset by Mishra & Soni,
2022 (Mendeley Data, CC BY 4.0, doi:10.17632/f45bkkt8pr.1); ASEA SMS Scam Detection dataset by
ChrisThePCGamer (GitHub, no license stated, used but not redistributed).

---

## Current results

*As of the 4,043-message core set (2,582 smishing, 1,461 legit) plus 5,289 training-only Mendeley
messages, cutoff 0.50. Stratified 5-fold cross-validation × 5 repeats (mean ± std over 25 folds).
These numbers change as the data grows; the app always shows the current ones.*

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.979 ± 0.004** | 0.972 ± 0.006 | **0.995 ± 0.002** | **0.983 ± 0.003** |
| Decision Tree | 0.969 ± 0.005 | 0.970 ± 0.006 | 0.982 ± 0.006 | 0.976 ± 0.004 |
| SVM | 0.973 ± 0.007 | **0.995 ± 0.003** | 0.964 ± 0.010 | 0.979 ± 0.005 |
| Random guess (baseline) | 0.491 ± 0.007 | 0.665 ± 0.010 | 0.408 ± 0.006 | 0.506 ± 0.007 |

- **All three models beat random guessing** (McNemar p < 0.001).
- **At cutoff 0.50, LR is significantly better than DT (p = 0.001).** LR vs SVM (p = 0.13) and
  DT vs SVM (p = 0.18) are not significantly different. The app recommends LR (highest recall).
- **The cutoff matters, especially for SVM.** SVM is the most cautious model (fewest false
  alarms). At 0.30, its recall rises from 0.964 to 0.982 and it becomes significantly better than
  both LR and DT (p < 0.001), with F1 0.987 ± 0.004.
- **Strongest hand-made features:** the sender features lead the permutation importance for
  all three models, most of all for the Decision Tree. Among the message features,
  `urgency_count`, `money_count` and `has_url` have the largest LR/SVM weights.
- **Hand-made features add real information (H3).** Compared with TF-IDF words alone, words +
  hand-made features are significantly better for all three models (McNemar p ≤ 0.005; F1 LR
  0.966 → 0.983, DT 0.904 → 0.976, SVM 0.972 → 0.979). Words + hand-made is also significantly
  better than hand-made alone for LR and SVM; for the Decision Tree the two are not significantly
  different (p = 0.50), since its splits are almost all on hand-made features. Note that the
  hand-made set includes the sender features, which partly reflect which dataset a message came from.
- **Per source:** on the held-out test set, LR scores F1 0.95 on `sms_taglish.csv`, which has no
  senders. The Decision Tree does worst there (12 false alarms in 40), since it relies most on
  the sender.

---

## Project structure

```
app.py                  Streamlit app: the UI and cached training
model.py                Training, evaluation, cross-validation and statistics (no UI)
features.py             Text cleaning and the 11 hand-made features
data_loader.py          Loads, standardises and deduplicates CSV/Excel files from datasets/
sms_taglish.csv         Our own Filipino/Taglish messages (part of the core set)
datasets/               Extra datasets and config.json (see datasets/README.md)
tests/test_features.py  Unit tests for features and the loader
tests/test_model.py     Tests for training/evaluation, plus a smoke test of the app
TEAM_GUIDE.md           Who does what next
requirements.txt        Pinned app dependencies
requirements-dev.txt    App dependencies + pytest
Chapter 1 & 2_Team TRIOjan_Virus.pdf   Study paper (Chapters 1–2)
```

---

## Limitations

From the study's Scope and Limitations:

- **It only reads the message.** It can't check whether a link is actually harmful, verify a
  sender with the telcos, or analyse images, attachments or calls.
- **It targets banking and online-payment smishing** in Filipino, English and Taglish. It isn't
  meant for general spam or other scams, and it doesn't handle other Philippine languages or dialects.
- **Results depend on the dataset.** Real SMS are hard to collect for privacy reasons, so the data
  may not fully match what people actually receive.
- **It doesn't use deep learning or transformer models.**
- **It is a proof of concept.** It is not a live filter and isn't connected to any telco or bank.
- **Scammers keep changing their wording**, so the models need retraining with new examples over time.
