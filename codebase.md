Project Path: smishguard

Source Tree:

```txt
smishguard
├── Chapter 1 & 2_Team TRIOjan_Virus (1).pdf
├── Chapter 1 & 2_Team TRIOjan_Virus.pdf
├── README.md
├── TEAM_GUIDE.md
├── app.py
├── data_loader.py
├── datasets
│   ├── Dataset_5971_(mendeley).csv
│   ├── README.md
│   ├── config.example.json
│   ├── config.json
│   └── tagalog-sms.xlsx
├── features.py
├── model.py
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
├── sms_taglish.csv
└── tests
    ├── test_features.py
    └── test_model.py

```

`README.md`:

```md
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

```
`TEAM_GUIDE.md`:

```md
# SmishGuard team guide

Who does what next, in order. Read sections 1 and 2 first, then your own part of section 3.

| Role | Who | Main job |
|---|---|---|
| **Lead** | Jakob | Reviews and merges everyone's work, runs the final numbers, coordinates the paper |
| **Admin system** | Ellise | Builds the admin system that reviews flagged messages |
| **Data & features** | Jel | Checks label quality, improves the features, prepares the results for the paper |

Everyone writes Chapter 3 together (section 4 lists what it has to cover).

---

## 1. Setup (everyone, do this first)

You need **Python 3.12 or newer** (`python --version`).

```powershell
git clone <repo url>
cd smishguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Then **download the ASEA dataset**, which can't be shared in our repo because it has no license:

1. Open <https://github.com/ChrisThePCGamer/asea-sms-scam-detection>.
2. Download `Final_Dataset/Combined-Dataset-V3(Sheet).csv`.
3. Save it in our `datasets/` folder as **`asea_sms_scam_detection_raw_dataset.csv`** (exactly this name).

Check that everything works:

```powershell
pytest                  # all tests should pass
streamlit run app.py    # opens http://localhost:8501
```

In the app's **Dataset** tab, the core set should show **4,043** messages. If it shows about
1,970, the ASEA file is missing or misnamed.

Read [README.md](README.md) (how the app works) and [datasets/README.md](datasets/README.md)
(where the data comes from and how it's labelled) before you start.

## 2. How we work

- **Only the lead commits to `main`.** Work on your own branch (`git checkout -b admin-system`,
  `git checkout -b keyword-lists`, …) and open a pull request, or send your changed files to the
  lead. The lead reviews, tests and merges.
- **Keep changes small and focused.** One task per branch is easier to review.
- **Run `pytest` before you hand anything in.** If you change features or data, also run
  cross-validation in the app and note the before/after numbers in your pull request.
- **Never commit dataset files** that `.gitignore` lists, and never commit real people's messages
  or phone numbers without their consent (Data Privacy Act of 2012).
- **Don't change these without asking the lead:** the label mapping in `datasets/config.json`,
  the test-set rules (`fit`, `add_extra`, `strata` in `model.py`), and the study's scope
  (banking/online-payment and gambling smishing in Filipino/Taglish SMS).

---

## 3. Tasks, in order

### Admin system (Ellise)

The goal: flagged messages get reviewed by a person before the models are trained on them.

**How flags work today.** In the Live demo tab, the 🚩/✅ buttons store a message, its sender and
the corrected label in memory (`flag_store()` in `app.py`). They're shared by everyone using the
app and **lost when it restarts**. The Feedback tab lists them, can download them as
`flagged.csv` (`text, label, sender`), and has a **Retrain with flagged messages** button, which
must stay.

**A1. Agree on the storage format with the lead (first, before any code).** A local CSV is enough
for the defense (no database). Suggested `feedback/flags.csv`:

| Column | Meaning |
|---|---|
| `id` | Unique number |
| `text`, `sender` | The flagged message |
| `label` | The label the user suggested (1 = smishing, 0 = legit) |
| `status` | `pending`, `approved` or `rejected` |
| `flagged_at` | When it was flagged |
| `reviewed_by`, `reviewed_at`, `note` | Filled in by the admin |

**A2. Save flags to that file.** When a 🚩/✅ button is pressed, append a `pending` row instead
of (or as well as) keeping it in memory. Flags then survive restarts.

**A3. Build the review screen.** A separate Streamlit page (e.g. `pages/admin.py`, which
Streamlit shows in the sidebar automatically) that:
- lists pending flags, with approve/reject buttons and an optional note;
- is protected by a password kept in `.streamlit/secrets.toml` (already in `.gitignore`);
- warns when a flag is a copy of a test-set message. The Feedback tab already does this check;
  see `in_test` in `app.py`. Training on test messages inflates the scores.

**A4. Feed approved flags into training.** Easiest, with no change to the training code: write
approved rows to `datasets/approved_flags.csv` (`text, label, sender`) and add it to
`datasets/config.json` with `"role": "train"`. They're then used for training only, never for
testing. Agree with the lead whether the Retrain button should also use only approved flags.

**Done when:** a flag survives an app restart, can be approved or rejected on the admin page,
and only approved flags reach training. Add tests for the CSV reading and writing to
`tests/`.

### Data & features (Jel)

You can start these in any order. D1 and D4 need little coding.

**D1. Label audit (for the paper's data-quality section).**
1. In the app's Dataset tab, or with pandas, take a random sample of about 300 core messages:
   about 100 from each source, half smishing and half legit.
2. Check every label against our definition: smishing = banking, e-wallet, prize, raffle, loan,
   investment, pasaload, delivery, government-spoofing or gambling scams; legit = everyday,
   bank/e-wallet notifications, OTPs and government notices.
3. Hand in a CSV with `source, text, given_label, your_label, note`, plus the error rate per
   source. Known suspects: ASEA's `HAM` includes English chit-chat, and some ASEA "Bank Scam"
   rows look like real bank ads.

**D2. Improve the keyword lists** (Objective 3b and 3c). `URGENT` and `MONEY` in
[features.py](features.py) are short. Look through the smishing messages for common Filipino and
Taglish words (e.g. *libre*, *deposito*, *taya*, *bonus*, *i-claim*, *mag-register*,
*na-lock*, *i-verify*) and add the useful ones.
- Add tests in `tests/test_features.py` like the existing `test_keyword_counts`.
- Compare cross-validation before and after (same cutoff, 5 repeats), and keep the change only if
  recall or F1 improves.

**D3. Handle scrambled spelling** (Objective 2). Scammers write *B0ss*, *pan@lunin* and *Sl0ts*
to dodge filters. Right now `clean()` turns the digits into `NUM`, so *Sl0ts* becomes `sl NUM ts`
and the word is lost.
- Try mapping 0→o, 1→i, 3→e, 4→a, @→a inside words that are mostly letters, before the digit
  step. Then consider a new hand-made feature that counts scrambled words, a likely smishing sign.
- New features go in `HAND_NAMES` and `hand()`; the tests show how.
- Compare cross-validation before and after, as in D2.

**D4. Results package for Chapter 4.** Once the lead freezes the data and code (section 5),
collect from the app the tables and charts the paper needs:
- Cross-validation table (cutoff 0.50 and the suggested cutoff) and the McNemar table (RQ1, RQ2, H1, H2).
- Feature-set comparison, hand-made feature ranks and permutation importance (RQ3, H3).
- Recall vs false-alarm chart (H4), test results by source, and the recommendation box (Objective 7).
- The Dataset tab counts (Objective 1).

Save them in one shared folder with the cutoff and date for each.

### Lead (Jakob)

- **L1.** Review and merge pull requests. Run `pytest` and cross-validation on each one.
- **L2.** Decide with Ellise the storage format (A1) and whether Retrain uses
  only approved flags (A4).
- **L3.** Decide whether to collect more of our own Filipino/Taglish messages. If yes, Jel can help: collect only with consent, remove names and numbers, add
  rows to `sms_taglish.csv`.
- **L4.** When the tasks are merged, freeze and update the README's results (section 5).

---

## 4. What Chapter 3 has to cover

The code makes choices the paper must describe. All are explained in the README and
`datasets/README.md`:

1. **Datasets:** the five sources, their licenses, which three are used, and why two are skipped.
2. **Label mapping:** which categories count as smishing, legit or dropped, and why.
3. **Core vs training-only data:** the core Filipino/Taglish set is split and cross-validated;
   the English Mendeley set is used for training only.
4. **Deduplication by template** (same message, different OTP/amount/link counts once), and
   dropping templates labelled both ways.
5. **Preprocessing:** lowercasing, links → `URL` + their words, digits → `NUM`, repairing
   garbled characters.
6. **Features:** TF-IDF of words and 2-word phrases, plus the 11 hand-made features.
7. **Sender dropout:** why every training message with a sender is also added without it.
8. **Models:** LR, DT (max depth 8), linear SVM (`LinearSVC`, calibrated), all class-balanced,
   plus the random-guess baseline.
9. **Evaluation:** stratified 80/20 split, 5-fold × 5 repeated cross-validation, accuracy,
   precision, recall, F1, McNemar's test, the feature-set comparison, and the cutoff analysis
   (F2 for the suggested cutoff).

## 5. Freeze, then write results

Once A1–A4 and D1–D3 are merged (or dropped), the lead:

1. Stops changing data and features.
2. Runs cross-validation and the feature-set comparison with 5 repeats.
3. Updates "Current results" in the README.

D4 (results package) and Chapter 4 then use those numbers. After the freeze, changing data or
features means redoing the results.

## 6. Defense prep (everyone)

- Prepare 5–6 demo messages for the Live demo: a GCash/BDO scam, a gambling scam, a real OTP, a
  real bank notification, casual Taglish, and one the models get wrong (shows the limits honestly).
- Rehearse the flag → admin review → retrain flow.
- Each member should be able to explain their own part, and the limitations in the README.

```
`app.py`:

```py
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import export_text
from sklearn.metrics import precision_score, recall_score

import data_loader
import model
from features import HAND_NAMES, sender_type
from model import (BASELINE, FEATURE_SETS, MODEL_NAMES, cv_significance, cv_summary, feature_set_table,
                   fit, metrics_table, predict_all, recommend)

st.set_page_config(page_title="SmishGuard", page_icon="🛡️", layout="wide")

# ---------- Cached training ----------
HERE = Path(__file__).parent
SOURCE_FILES = [Path(__file__), HERE / "model.py", HERE / "features.py", HERE / "data_loader.py"]


def cache_key():
    """Key for every cached result. It changes when the code, sms_taglish.csv or anything in datasets/
    changes, so a running server (e.g. Streamlit Cloud after a git push) never mixes models trained
    by old code or old data with the current code."""
    h = hashlib.sha1()
    for f in SOURCE_FILES + [model.BASE_CSV]:
        h.update(f.read_bytes() if f.exists() else b"")
    return (h.hexdigest()[:12], data_loader.signature())


@st.cache_resource(show_spinner="Loading the datasets...")
def load_data(sig):
    """`sig` (from cache_key) changes when the code or data change, which reloads everything.
    Returns (core, train_extra, per-file summary, n_conflicts); see model.load_data."""
    return model.load_data()


@st.cache_resource(show_spinner="Training models...")
def base_artifacts(sig):
    core, train_extra, _, _ = load_data(sig)
    return fit(core, train_extra)


@st.cache_resource(show_spinner="Running cross-validation (this can take a while with big datasets)...")
def run_cv(sig, n_repeats, features="both"):
    core, train_extra, _, _ = load_data(sig)
    return model.cross_validate(core, train_extra, n_repeats, features=features)


@st.cache_resource
def flag_store():
    """Shared in-memory list of flagged messages (all visitors share it).
    Lost when the app restarts, so download the CSV to keep it."""
    return {}


# ---------- UI ----------
st.title("🛡️ SmishGuard")
st.caption("Detecting banking and online payment smishing in Filipino/Taglish SMS")

sig = cache_key()
try:
    base_art = base_artifacts(sig)
except FileNotFoundError:
    st.error("`sms_taglish.csv` not found. Put it in the same folder as app.py.")
    st.stop()

# Use the retrained model if this session has one
if st.session_state.get("retrained_sig", sig) != sig:
    st.session_state.pop("art_retrained", None)  # code or data changed: drop the stale retrain
art = st.session_state.get("art_retrained", base_art)
flags = flag_store()

with st.sidebar:
    st.header("Settings")
    cutoff = st.slider("Smishing cutoff", 0.05, 0.95, 0.50, 0.05,
                       help="A message is called SMISHING when a model's smishing probability is at or above "
                            "this value. Lower it to catch more scams (higher recall) at the cost of more false alarms.")
    st.caption("Used by the Live demo and the Model comparison tab.")

tab_demo, tab_fb, tab_models, tab_feat, tab_data = st.tabs(
    ["🔎 Live demo", "🚩 Feedback", "📊 Model comparison", "🧠 Top features", "🗂️ Dataset"])

# (sender, message)
EXAMPLES = {
    "GCash scare + short link": ("+639561234567", "GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify"),
    "Prize scam": ("09281234567", "Congrats! Nanalo ka ng P30,000. I-claim now: promo-ph.xyz"),
    "Real bank OTP": ("BDO", "Your BDO OTP is 482913. Do not share this code with anyone."),
    "Casual Taglish chat": ("", "pre nasend ko na yung pambayad, check mo na lang"),
}

with tab_demo:
    if "art_retrained" in st.session_state:
        st.info("Using the **retrained** model (includes flagged messages).")
    choice = st.selectbox("Try an example (or write your own below)", ["— custom —"] + list(EXAMPLES))
    default_sender, default = EXAMPLES.get(choice, ("", ""))
    sender = st.text_input("Sender (optional): a number like 09171234567 or a name like BDO",
                           value=default_sender, key=f"sender_{choice}")
    msg = st.text_area("Paste an SMS message", value=default, height=120, key=f"msg_{choice}")

    if st.button("Analyze", type="primary") and msg.strip():
        st.session_state["last"] = {"msg": msg.strip(), "sender": sender.strip(),
                                    "probs": predict_all(art, msg, sender)}

    last = st.session_state.get("last")
    if last:
        probs = last["probs"]
        st.markdown(f"**Sender:** {last['sender'] or '(not given)'} ({sender_type(last['sender'])})  \n"
                    f"**Message:** {last['msg']}")
        cols = st.columns(3)
        for col, (name, p) in zip(cols, probs.items()):
            with col:
                st.subheader(name)
                if p >= cutoff:
                    st.error(f"⚠️ SMISHING ({p:.0%})")
                else:
                    st.success(f"✅ Legit ({p:.0%} smishing)")
                st.progress(min(max(p, 0.0), 1.0))
        votes = sum(p >= cutoff for p in probs.values())
        st.info(f"{votes} of 3 models flag this message as smishing (cutoff {cutoff:.2f}).")

        st.markdown("**Is the result wrong? Tell SmishGuard the correct label:**")
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("🚩 It's actually SMISHING"):
            flags[(last["msg"], last["sender"])] = 1
            st.toast("Flagged as smishing. See the Feedback tab.")
        if b2.button("✅ It's actually LEGIT"):
            flags[(last["msg"], last["sender"])] = 0
            st.toast("Flagged as legit. See the Feedback tab.")

with tab_fb:
    st.write("Messages you flag are collected here. Retrain to teach the models from them.")
    if not flags:
        st.info("Nothing flagged yet. Analyze a message in the Live demo tab, then use the flag buttons.")
    else:
        fdf = pd.DataFrame({"text": [m for m, _ in flags], "label": list(flags.values()),
                            "sender": [snd for _, snd in flags]})
        in_test = fdf.text.map(data_loader.dedup_key).isin(base_art["test_keys"])
        shown = fdf.assign(label=fdf.label.map({1: "Smishing", 0: "Legit"}),
                           **{"In test set": in_test.map({True: "Yes (not used)", False: ""})})
        st.dataframe(shown, width="stretch")
        if in_test.any():
            st.warning(f"{int(in_test.sum())} flagged message(s) are in the held-out test set, so retraining "
                       "leaves them out on purpose: training on test messages would make the scores look better "
                       "than they really are. Rephrase the message slightly if you want the model to learn from it.")

        repeat = st.slider("Repeat each flagged message N times when training", 1, 20, 5,
                           help="One extra message barely moves a model trained on hundreds of rows, so repeating it gives it more weight.")
        c1, c2, c3 = st.columns(3)
        if c1.button("🔁 Retrain with flagged messages", type="primary"):
            core, train_extra, _, _ = load_data(sig)
            extra = pd.concat([train_extra] + [fdf] * repeat, ignore_index=True)
            with st.spinner("Retraining..."):
                st.session_state["art_retrained"] = fit(core, extra)
                st.session_state["retrained_sig"] = sig
            st.success("Retrained! The Live demo now uses the updated models. Re-analyze your message to check.")
            st.rerun()
        c2.download_button("⬇️ Download flagged.csv", fdf.to_csv(index=False),
                           file_name="flagged.csv", mime="text/csv")
        if c3.button("🗑️ Clear flags"):
            flags.clear()
            st.rerun()

    if "art_retrained" in st.session_state:
        if st.button("↩️ Go back to the original model"):
            del st.session_state["art_retrained"]
            st.rerun()

    st.caption("Heads up: flags are shared by everyone using the app and are lost when the app restarts. "
               "To keep them permanently, download flagged.csv and add its rows to sms_taglish.csv in GitHub.")

with tab_models:
    # --- Single held-out split ---
    st.subheader("Held-out test set")
    st.write(f"Train: **{art['n_train']}** messages (incl. **{art['n_extra']}** training-only rows from "
             f"datasets/ and flags) | Test: **{art['n_test']}** held-out core Filipino/Taglish messages | "
             f"Cutoff: **{cutoff:.2f}**")
    results = metrics_table(art["y_test"], art["test_probs"], cutoff)
    st.dataframe(results, width="stretch")
    st.bar_chart(results[["Accuracy", "Precision", "Recall", "F1"]], stack=False)
    st.caption("Missed scams (FN) are the costly error here: a smishing text that got through. "
               f"With {art['n_test']} test messages, one message moves accuracy by "
               f"{1 / art['n_test']:.1%}; use the cross-validation below for the paper.")

    st.markdown("**Test results by source** (the same test set, split by the file each message came from)")
    src = art["test_source"]
    by_src = pd.concat({s: metrics_table(art["y_test"][src == s],
                                         {n: p[src == s] for n, p in art["test_probs"].items()}, cutoff)
                        for s in pd.unique(src)}, names=["Source", "Model"])
    st.dataframe(by_src.drop(index=BASELINE, level="Model"), width="stretch")
    st.caption("sms_taglish.csv has no senders, while most messages in the other core files do. A big gap in "
               "recall between sources can mean the models lean on the sender features.")

    # --- Cutoff trade-off (H4) ---
    st.subheader("Recall vs false alarms by cutoff")
    pick = st.selectbox("Model", MODEL_NAMES, index=2, key="curve_model")
    y, p = art["y_test"], art["test_probs"][pick]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    curve = pd.DataFrame([{"Cutoff": c,
                           "Recall": recall_score(y, p >= c, zero_division=0),
                           "Precision": precision_score(y, p >= c, zero_division=0),
                           "False alarm rate": ((p >= c) & (y == 0)).sum() / max((y == 0).sum(), 1)}
                          for c in grid]).set_index("Cutoff")
    st.line_chart(curve)
    st.caption(f"Lower cutoffs catch more scams (recall up) but flag more legit texts (false alarm rate up). "
               f"Current cutoff {cutoff:.2f}: recall {curve.Recall.asof(cutoff):.3f}, "
               f"false alarm rate {curve['False alarm rate'].asof(cutoff):.3f}.")

    # --- Cross-validation (H1, H2) ---
    st.subheader("Cross-validation")
    st.write("Stratified 5-fold cross-validation on the Taglish data, repeated with different shuffles. "
             "Every message is tested once per repeat; `datasets/` files are added to the training folds only.")
    n_rep = st.select_slider("Repeats", [1, 3, 5, 10], value=5, key="cv_repeats")
    if st.button("▶️ Run cross-validation") or st.session_state.get("cv_ran") == (sig, n_rep):
        st.session_state["cv_ran"] = (sig, n_rep)
        cv = run_cv(sig, n_rep)
        table, mean = cv_summary(cv, cutoff)
        st.markdown(f"**Mean ± std over {n_rep * cv['n_splits']} folds** (cutoff {cutoff:.2f})")
        st.dataframe(table, width="stretch")

        st.markdown("**McNemar's test**: do two models make significantly different mistakes?")
        st.dataframe(cv_significance(cv, cutoff), width="stretch", hide_index=True)
        st.caption("Exact McNemar test on the out-of-fold predictions of each repeat (each message predicted once "
                   "per repeat); the median p-value across repeats is reported. p < 0.05 means the difference "
                   "is unlikely to be chance. Rows against the baseline test H1; rows between models test H2.")

        rec = recommend(cv, cutoff)
        b, r, m = rec["best"], rec["runner_up"], rec["mean"]
        lead = (f"and its lead over {r} is **statistically significant** (median p = {rec['p']:.4f})" if rec["p"] < 0.05
                else f"but its lead over {r} is **not statistically significant** (median p = {rec['p']:.4f}), "
                     "so either could be used")
        cs = rec["cut_summary"]
        st.subheader("Recommendation")
        st.success(
            f"**Deploy {b}.** At cutoff {cutoff:.2f} it has the highest cross-validated recall "
            f"({m.loc[b, 'Recall']:.3f}, F1 {m.loc[b, 'F1']:.3f}), {lead}.  \n"
            f"**Suggested cutoff: {rec['cutoff']:.2f}** (best F2 score, which weighs recall twice as much as "
            f"precision): recall {cs['Recall']:.3f}, precision {cs['Precision']:.3f}, accuracy {cs['Accuracy']:.3f}.  \n"
            f"Random guessing reaches only {m.loc[BASELINE, 'Accuracy']:.3f} accuracy.")
        st.caption("Calculated from the cross-validation above, so it updates when the data in sms_taglish.csv "
                   "or datasets/ changes. See the Top features tab for which features matter most.")
    else:
        st.info("Click **Run cross-validation** to compute it. The result is kept until the data changes.")

with tab_feat:
    imp, names = art["importance"], art["feature_names"]
    n_feat = len(names)

    # --- Hand-made features across all models (RQ3) ---
    st.subheader("Hand-made features across all 3 models")
    rows = {}
    for mname, series in imp.items():
        order = series.sort_values(ascending=False)
        rank = pd.Series(np.arange(1, n_feat + 1), index=order.index)
        label = "importance" if mname == "Decision Tree" else "weight"
        rows[f"{mname} {label}"] = series[HAND_NAMES].round(3)
        rows[f"{mname} rank"] = rank[HAND_NAMES].where(series[HAND_NAMES].abs() >= 1e-3)  # unused -> blank
    hand_tbl = pd.DataFrame(rows)
    hand_tbl["Average rank"] = hand_tbl[[c for c in hand_tbl if c.endswith("rank")]].mean(axis=1).round(1)
    st.dataframe(hand_tbl.sort_values("Average rank"), width="stretch")
    st.caption(f"Rank = position among all {n_feat:,} features (word/phrase TF-IDF + {len(HAND_NAMES)} hand-made), "
               "1 = strongest push toward SMISHING. LR/SVM weights are signed (negative = pushes toward legit); "
               "Decision Tree importance is unsigned (how much the tree relies on it). Blank rank = the model "
               "doesn't use that feature at all. sms_taglish.csv has no senders, so for its messages only "
               "`sender_unknown` is ever set.")

    # --- Permutation importance (model-agnostic, from CV) ---
    st.subheader("Permutation importance (from cross-validation)")
    n_rep = st.session_state.get("cv_repeats", 5)
    if st.session_state.get("cv_ran") == (sig, n_rep):
        perm = run_cv(sig, n_rep)["perm"]
        metric = st.radio("Measure", ["Recall drop", "Accuracy drop"], horizontal=True)
        ptbl = perm.pivot(index="Feature", columns="Model", values=metric)[MODEL_NAMES]
        ptbl = ptbl.loc[ptbl.mean(axis=1).sort_values(ascending=False).index]
        st.bar_chart(ptbl, horizontal=True, stack=False)
        st.dataframe(ptbl.round(4), width="stretch")
        st.caption("How much each model's score drops when one hand-made feature is shuffled in the test folds "
                   "(5 shuffles x 5 folds). Bigger drop = the model relies on it more. Near zero = the model "
                   "doesn't need it, often because TF-IDF words already carry the same signal.")
    else:
        st.info("Run cross-validation in the **Model comparison** tab first; this section uses the same run.")

    # --- Feature sets (H3, Obj. 7) ---
    st.subheader("Feature sets: words, hand-made features, or both")
    st.write("Cross-validates every model again with only the TF-IDF words, and with only the 11 hand-made "
             f"features, on the same folds ({n_rep} repeats, as set in the Model comparison tab). "
             "McNemar's test then checks each against words + hand-made.")
    if st.button("▶️ Run feature-set comparison") or st.session_state.get("fs_ran") == (sig, n_rep):
        st.session_state["fs_ran"] = (sig, n_rep)
        cvs = {fs: run_cv(sig, n_rep, fs) for fs in FEATURE_SETS}
        st.dataframe(feature_set_table(cvs, cutoff), width="stretch")
        st.caption(f"Cutoff {cutoff:.2f}. \"Significantly worse\" for words only means the hand-made features "
                   "add real information on top of the words; \"Not significantly different\" means the words "
                   "already carry the same signal.")
    else:
        st.info("Click **Run feature-set comparison** to compute it (it runs cross-validation twice more).")

    # --- Each model's own view ---
    st.subheader("Top features per model")
    pick = st.selectbox("Model", MODEL_NAMES, key="feat_model")
    series = imp[pick]
    if pick == "Decision Tree":
        top = series.sort_values(ascending=False).head(15)
        st.dataframe(top.round(4).rename("importance"))
        st.markdown("**First 3 levels of the tree** (1 = smishing, 0 = legit; hand features are scaled to 0..1)")
        st.code(export_text(art["models"]["Decision Tree"], feature_names=names, max_depth=3,
                            show_weights=False), language="text")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pushes toward SMISHING**")
            st.dataframe(series.sort_values(ascending=False).head(15).round(3).rename("weight"))
        with c2:
            st.markdown("**Pushes toward LEGIT**")
            st.dataframe(series.sort_values().head(15).round(3).rename("weight"))
        top = series.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top[::-1].plot(kind="barh", color="#c0392b", ax=ax)
    ax.set_title(f"Top smishing indicators ({pick})")
    fig.tight_layout()
    st.pyplot(fig)

with tab_data:
    df = base_art["df"]
    _, train_extra, summary, n_conf = load_data(sig)
    st.subheader("Core Filipino/Taglish dataset")
    st.write(f"{len(df)} messages: `sms_taglish.csv` plus the `datasets/` files with role **core**, one row per "
             "template (the same message with a different OTP, amount or link counts once). The **Held-out test "
             "set** results use a fixed 80/20 split of it (the Live demo model is trained on the 80%). "
             "**Cross-validation** rotates through all of it, so every message is tested once per repeat.")
    counts = df.assign(label=df.label.map({0: "Legit", 1: "Smishing"})).groupby(["source", "label"]).size()
    st.dataframe(counts.unstack(fill_value=0), width="stretch")
    if n_conf:
        st.caption(f"{n_conf} rows were left out because the same template was labelled both smishing and legit.")
    st.dataframe(df.sample(min(10, len(df)), random_state=1), width="stretch")

    st.subheader("datasets/ folder")
    if summary.empty:
        st.info("No CSV or Excel files in `datasets/` yet. Drop Kaggle files there and refresh the page; "
                "see `datasets/README.md` for the format.")
    else:
        st.write(f"**{(summary.Status == 'OK').sum()}** file(s) loaded. Role **core** files join the split above; "
                 f"role **train** files add **{len(train_extra)}** unique messages to training only.")
        st.dataframe(summary, width="stretch", hide_index=True)
        if (~summary.Status.eq("OK") & ~summary.Status.str.startswith("Skipped: skipped in config")).any():
            st.warning("Some files were skipped. See the Status column, and `datasets/config.json` "
                       "(`datasets/README.md` explains it).")

```
`data_loader.py`:

```py
"""Loads the extra datasets in datasets/ (CSV or Excel, e.g. from Kaggle) into one text/label table.

Each file is mapped to two columns: `text` and `label` (1 = smishing/spam, 0 = legit).
Columns are auto-detected from common names. If that guesses wrong, add an entry for the
file in datasets/config.json (see datasets/README.md).
"""
import json
import re
from pathlib import Path

import pandas as pd

from features import URL_RE, clean

DATA_DIR = Path(__file__).parent / "datasets"
PATTERNS = ("*.csv", "*.xlsx")

TEXT_COLS = ["text", "message", "sms", "msg", "v2", "content", "body", "sms_text", "message_text"]
# Not "phone": in some sets (e.g. Mendeley) PHONE is a yes/no flag for "message contains a number"
SENDER_COLS = ["sender", "from", "sender_id", "address", "number", "masked_celphone_number"]
LABEL_COLS = ["label", "v1", "category", "class", "type", "target", "spam", "is_spam", "labels", "intent"]
POSITIVE = {"1", "spam", "smishing", "smish", "scam", "phishing", "fraud", "true", "yes"}
NEGATIVE = {"0", "ham", "legit", "legitimate", "normal", "not spam", "false", "no"}
ROLES = {"train", "core"}


def _read(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    # Many Kaggle SMS files (e.g. UCI spam.csv) are latin-1, not UTF-8
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode file")


def fix_mojibake(t):
    """Repair UTF-8 text that was decoded as Windows-1252 ("Weâ€™re" -> "We’re"). Text that
    doesn't round-trip cleanly is left as it is."""
    if not re.search(r"[ÂÃâð]", t):
        return t
    try:
        return t.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return t


def dedup_key(text):
    """Messages from one template (same words, different OTP, amount or link) share a key."""
    return re.sub(r"\s+", " ", clean(URL_RE.sub(" URL ", str(text)))).strip()


def dedupe(df):
    """Keep one row per template. A template labelled both smishing and legit is dropped,
    since one of the labels must be wrong. Returns (deduped df, number of conflicting rows)."""
    key = df.text.map(dedup_key)
    conflict = df.groupby(key).label.transform("nunique") > 1
    out = df[~conflict & ~key.duplicated()]
    return out.reset_index(drop=True), int(conflict.sum())


def _find(cols, candidates):
    lower = {str(c).strip().lower(): c for c in cols}
    return next((lower[c] for c in candidates if c in lower), None)


def _to_label(series, positive=None, negative=None):
    pos = {str(v).strip().lower() for v in positive} if positive else POSITIVE
    neg = {str(v).strip().lower() for v in negative} if negative else NEGATIVE
    s = series.astype(str).str.strip().str.lower()
    # Numeric labels like 1.0 / 0.0
    s = s.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out[s.isin(pos)] = 1
    if negative or not positive:
        out[s.isin(neg)] = 0
    else:
        out[~s.isin(pos)] = 0  # only `positive` given: everything else is legit
    return out


def _files(data_dir):
    return sorted(f for p in PATTERNS for f in Path(data_dir).glob(p) if not f.name.startswith("~$"))


def load_config(data_dir=DATA_DIR):
    config_file = Path(data_dir) / "config.json"
    if config_file.exists():
        return json.loads(config_file.read_text(encoding="utf-8"))
    return {}


def signature(data_dir=DATA_DIR):
    """Changes whenever a data file or the config is added, removed or edited (used as a cache key)."""
    data_dir, config_file = Path(data_dir), Path(data_dir) / "config.json"
    if not data_dir.exists():
        return ()
    files = _files(data_dir) + ([config_file] if config_file.exists() else [])
    return tuple((f.name, f.stat().st_size, f.stat().st_mtime) for f in files)


def load_extra(data_dir=DATA_DIR):
    """Returns (combined DataFrame[text, label, sender, source, role], per-file summary DataFrame).
    role is "train" (training only, the default) or "core" (joins sms_taglish.csv in the
    train/test split, so it is also tested on). Rows are not deduplicated here; see dedupe()."""
    data_dir = Path(data_dir)
    config = load_config(data_dir)
    frames, summary = [], []
    for path in _files(data_dir) if data_dir.exists() else []:
        cfg = config.get(path.name, {})
        row = {"File": path.name, "Role": cfg.get("role", "train"), "Text column": None,
               "Label column": None, "Sender column": None,
               "Rows used": 0, "Smishing": 0, "Legit": 0, "Dropped": 0, "Status": "OK"}
        try:
            if cfg.get("skip"):
                raise ValueError("skipped in config.json" + (f" ({cfg['note']})" if cfg.get("note") else ""))
            if row["Role"] not in ROLES:
                raise ValueError(f"role must be one of {sorted(ROLES)} in config.json")
            df = _read(path)
            tcol = cfg.get("text") or _find(df.columns, TEXT_COLS)
            lcol = cfg.get("label") or _find(df.columns, LABEL_COLS)
            if tcol not in df.columns or lcol not in df.columns:
                raise ValueError(f"can't find text/label columns in {list(df.columns)[:8]}; "
                                 "set them in config.json")
            scol = cfg.get("sender") or _find(df.columns, SENDER_COLS)
            row["Text column"], row["Label column"], row["Sender column"] = tcol, lcol, scol
            out = pd.DataFrame({"text": df[tcol].fillna("").astype(str).str.strip().map(fix_mojibake),
                                "label": _to_label(df[lcol], cfg.get("positive"), cfg.get("negative")),
                                "sender": df[scol].fillna("").astype(str).str.strip() if scol in df else ""})
            before = len(out)
            out = out[(out.text != "") & (out.text.str.lower() != "nan")].dropna(subset=["label"])
            out["label"] = out.label.astype(int)
            out["source"], out["role"] = path.name, row["Role"]
            row.update({"Rows used": len(out), "Smishing": int(out.label.sum()),
                        "Legit": int((out.label == 0).sum()), "Dropped": before - len(out)})
            if out.empty:
                raise ValueError("no usable rows (check the label values in config.json)")
            frames.append(out)
        except Exception as e:  # one bad file shouldn't break the app
            row["Status"] = f"Skipped: {e}"
        summary.append(row)

    extra = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame({"text": [], "label": [], "sender": [], "source": [], "role": []}))
    return extra, pd.DataFrame(summary)

```
`datasets\Dataset_5971_(mendeley).csv`:

```csv
CSV Schema (1 sample row):
Headers: LABEL, TEXT, URL, EMAIL, PHONE
Sample: "ham", "Your opinion about me? 1. Over 2. Jada 3. Kusruthi 4. Lovable 5. Silent 6. Spl character 7. Not matured 8. Stylish 9. Simple Pls reply..", "No", "No", "No"
... [5970 more rows omitted]

```
`datasets\README.md`:

```md
# datasets/

Put extra datasets here (for example from Kaggle). Every `.csv` and `.xlsx` in this folder is
loaded automatically when the app starts or the page is refreshed.

Each file has a **role**, set in `config.json`:

- `train` (the default): used for **training only**. Good for English sets like Mendeley.
- `core`: joins `../sms_taglish.csv` as part of the **core Filipino/Taglish set**. The core set
  is what gets split 80/20 and cross-validated, so these messages are tested on too.

Other rules:

- Messages are deduplicated by **template**: the same message with a different OTP, amount or
  link counts once. A template labelled both smishing and legit is dropped.
- Training rows whose template also appears in the test set are dropped automatically.
- Text decoded with the wrong encoding (`Weâ€™re`) is repaired where possible.
- The **Dataset** tab in the app lists each file, its role, how many rows were used, and why a
  file was skipped.

## Files in use

| File | Role | Smishing (1) | Legit (0) | Dropped |
|---|---|---|---|---|
| `asea_sms_scam_detection_raw_dataset.csv` | core | Bank, Banking, Prize, Raffle, Loan, Investment, Pasaload, Delivery, Government Spoofing and Gambling scams | `HAM` | Commercial Spam, Job Scam, Other, spam |
| `tagalog-sms.xlsx` | core | `spam` | `notifs`, `otp`, `gov` | `ads` |
| `Dataset_5971_(mendeley).csv` | train | `smishing` | `ham` | `spam` (general spam) |
| `bwandowando_sms_scam_dataset.csv` | skipped | | | No labels; ~80% is already in ASEA |
| `AGR-Yes_SPAM_SMS_dataset.csv` | skipped | | | No labels; ~91% is already in ASEA |

## Getting the files, and licenses

Only the files whose license allows sharing are in the repo. Download the others and save them
here **under exactly the file name shown**, or `config.json` won't match them.

| File | In repo? | Source | License |
|---|---|---|---|
| `tagalog-sms.xlsx` | Yes | [Kaggle: onzero0/tagalog-sms](https://www.kaggle.com/datasets/onzero0/tagalog-sms) | CC0 (public domain) |
| `Dataset_5971_(mendeley).csv` | Yes | [Mendeley Data: f45bkkt8pr](https://data.mendeley.com/datasets/f45bkkt8pr) (Mishra & Soni, 2022) | CC BY 4.0 (credit required) |
| `asea_sms_scam_detection_raw_dataset.csv` | **No**, download it | [GitHub: ChrisThePCGamer/asea-sms-scam-detection](https://github.com/ChrisThePCGamer/asea-sms-scam-detection), file `Final_Dataset/Combined-Dataset-V3(Sheet).csv`, renamed | None stated (all rights reserved) |
| `bwandowando_sms_scam_dataset.csv` | No (not used) | [Kaggle: bwandowando/philippine-spam-sms-messages](https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages) | CC BY-NC-SA 4.0 |
| `AGR-Yes_SPAM_SMS_dataset.csv` | No (not used) | [GitHub: AGR-Yes/ScamMessagesPhilippines](https://github.com/AGR-Yes/ScamMessagesPhilippines) | None stated |

Files with no license can still be used for the study, but not republished, so they're listed in
`../.gitignore`. Cite every source used in the paper. Without the ASEA file, the app still runs,
but the core set is smaller and the scores won't match the README.

## Accepted format

Each file needs one text column and one label column. Common names are detected for you:

| | Detected column names |
|---|---|
| Text | `text`, `message`, `sms`, `msg`, `v2`, `content`, `body`, `sms_text`, `message_text` |
| Label | `label`, `v1`, `category`, `class`, `type`, `target`, `spam`, `is_spam`, `labels`, `intent` |
| Sender (optional) | `sender`, `from`, `sender_id`, `address`, `number`, `masked_celphone_number` |

A column named `phone` is not used as the sender, since in some sets (e.g. Mendeley) it is a
yes/no flag. Name it in `config.json` if it really holds senders.

Label values are converted to `1` (smishing/scam) or `0` (legit):

- `1`: `1`, `spam`, `smishing`, `smish`, `scam`, `phishing`, `fraud`, `true`, `yes`
- `0`: `0`, `ham`, `legit`, `legitimate`, `normal`, `not spam`, `false`, `no`

Rows with any other label are dropped (shown in the "Dropped" column).

The sender column is optional. A sender that is a mobile number (`09171234567`, `+639...`, or a
masked one like `+63960****655`) counts as `number`; a name or short code (`BDO`, `GCash`, `2958`)
counts as `alpha`; blank counts as `unknown`. Files without a sender column are treated as all
`unknown`. Numbers are not checked against a list of network prefixes: the 2023 list misses
prefixes such as 0981, 0985 and 0900, which are common in these datasets.

## config.json

Keyed by file name. Every field is optional.

```json
{
  "kaggle_smishing.csv": {
    "role": "train",
    "text": "TEXT",
    "label": "LABEL",
    "positive": ["smishing"],
    "negative": ["ham"]
  },
  "big_english_spam.csv": { "skip": true, "note": "general spam, out of scope" }
}
```

- `role`: `train` (default) or `core`, as described above.
- `text` / `label` / `sender`: the exact column names in that file (`sender` is optional).
- `positive`: the label values that count as smishing (`1`).
- `negative`: the label values that count as legit (`0`). If you give only `positive`,
  every other value counts as legit. If you give both, every other value is dropped.
- `skip`: `true` ignores the file without deleting it.
- `note`: free text, shown in the Dataset tab when the file is skipped.

Tip: some Kaggle sets have three classes, like `ham` / `spam` / `smishing`. Our paper only
targets smishing, so to keep general spam out, use `"positive": ["smishing"], "negative": ["ham"]`
(rows labelled `spam` are then dropped).

```
`datasets\config.example.json`:

```json
{
  "kaggle_smishing.csv": {
    "role": "train",
    "text": "TEXT",
    "label": "LABEL",
    "positive": ["smishing"],
    "negative": ["ham"]
  },
  "big_english_spam.csv": { "skip": true }
}

```
`datasets\config.json`:

```json
{
  "asea_sms_scam_detection_raw_dataset.csv": {
    "role": "core",
    "text": "text",
    "label": "intent",
    "sender": "masked_celphone_number",
    "positive": ["Bank Scam", "Banking Scam", "Prize Scam", "Raffle Scam", "Loan Scam", "Investment Scam",
                 "Pasaload Scam", "Delivery Scam", "Government Spoofing", "Gambling Scam"],
    "negative": ["HAM"],
    "note": "Commercial Spam, Job Scam, Other and spam are dropped (out of scope or ambiguous)"
  },
  "tagalog-sms.xlsx": {
    "role": "core",
    "text": "text",
    "label": "category",
    "sender": "sender",
    "positive": ["spam"],
    "negative": ["notifs", "otp", "gov"],
    "note": "ads are dropped (official promos, neither smishing nor everyday legit SMS)"
  },
  "Dataset_5971_(mendeley).csv": {
    "role": "train",
    "positive": ["smishing"],
    "negative": ["ham"],
    "note": "English; general spam is dropped"
  },
  "bwandowando_sms_scam_dataset.csv": {
    "skip": true,
    "note": "no labels, and ~80% is already in the ASEA file with labels"
  },
  "AGR-Yes_SPAM_SMS_dataset.csv": {
    "skip": true,
    "note": "no labels, and ~91% is already in the ASEA file with labels"
  }
}

```
`features.py`:

```py
"""Text cleaning and hand-made features for SmishGuard (shared by the app and the tests)."""
import re

URGENT = ["kailangan agad", "agad", "ngayon", "claim now", "urgent", "expire", "last chance", "within 24"]
MONEY = ["php", "₱", "peso", "prize", "premyo", "panalo", "nanalo", "won", "reward", "cash", "gcash", "maya", "bdo", "bpi", "account"]
HAND_NAMES = ["has_url", "short_url", "suspicious_tld", "urgency_count", "money_count", "length",
              "special_chars", "has_phone", "sender_is_number", "sender_is_alpha", "sender_unknown"]
# PH mobile numbers: 09xx xxx xxxx, +639..., 639...
PH_PHONE = re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")

# Links: http(s)://..., www...., or a bare domain ending in a known TLD (so "P500.00" or "Ref: SM2210." don't count)
TLDS = ("com|net|org|ph|gov|edu|info|biz|io|co|me|ly|cc|xyz|top|online|site|click|link|live|shop|store|"
        "app|icu|buzz|vip|tk|ml|ga|cf|gq|gd|gy")
URL_RE = re.compile(rf"https?://\S+|www\.\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:{TLDS})\b(?:/\S*)?", re.I)
SHORTENERS = re.compile(r"\b(?:bit\.ly|tinyurl\.com|cutt\.ly|shorturl\.at|is\.gd|goo\.gl|rb\.gy|ow\.ly|tiny\.cc|s\.id|t\.me)\b", re.I)
# Cheap TLDs often used for throwaway scam domains (official PH sites use .com, .com.ph, .gov.ph)
SUSPICIOUS_TLD = re.compile(r"\.(?:xyz|top|cc|info|online|site|click|link|live|shop|icu|buzz|vip|tk|ml|ga|cf|gq)\b(?:/|$)", re.I)


def urls(t):
    return [m.group().rstrip(".,!?)") for m in URL_RE.finditer(t)]


def clean(t):
    # Replace each link with a URL token but keep its words ("gcash-verify.top" -> "URL gcash verify top"),
    # since words like "verify" inside scam domains are useful signals
    t = URL_RE.sub(lambda m: " URL " + re.sub(r"[^a-z0-9]+", " ", m.group().lower()) + " ", t.lower())
    return re.sub(r"\d+", " NUM ", t)


def sender_type(sender):
    """'number' (plain mobile number), 'alpha' (named ID like BDO or a short code like 2958) or 'unknown' (blank)."""
    s = str(sender or "").strip()
    if not s or s.lower() == "nan":
        return "unknown"
    # `*` covers masked numbers in public datasets ("+63960****655"). No prefix whitelist: the 2023
    # list of PH network prefixes misses newer ones (0981, 0985, 0900, ...) common in the data
    if re.fullmatch(r"\+?[\d\s*-]{7,}", s) and re.search(r"\d", s):
        return "number"
    return "alpha"


def hand(t, sender=""):
    l = t.lower()
    st_ = sender_type(sender)
    links = urls(t)
    return [
        int(bool(links)),
        int(any(SHORTENERS.search(u) for u in links)),
        int(any(SUSPICIOUS_TLD.search(u.split("?")[0]) for u in links)),
        sum(k in l for k in URGENT),
        sum(k in l for k in MONEY),
        len(t),
        sum(not c.isalnum() and not c.isspace() for c in t),
        int(bool(PH_PHONE.search(t))),
        int(st_ == "number"),
        int(st_ == "alpha"),
        int(st_ == "unknown"),
    ]

```
`model.py`:

```py
"""Training, evaluation and statistics for SmishGuard, kept apart from the Streamlit UI (app.py)
so it can be tested and run without the app."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix)

import data_loader
from features import HAND_NAMES, clean, hand

BASE_CSV = Path(__file__).parent / "sms_taglish.csv"

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "SVM"]
BASELINE = "Random guess (baseline)"
# Which features the models see (H3: do the hand-made features add anything over the words?)
FEATURE_SETS = {"both": "Words + hand-made", "words": "Words only (TF-IDF)", "hand": "Hand-made only"}


def make_models():
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        # Linear SVM. LinearSVC (liblinear) instead of SVC(kernel="linear") (libsvm): same kind of model,
        # ~50x faster on thousands of messages. Calibration gives it smishing probabilities
        "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), ensemble=False),
        # Guesses at random in the same smishing/legit ratio as the training data (H1 comparison)
        BASELINE: DummyClassifier(strategy="stratified", random_state=42),
    }


def add_extra(tr, te, extra):
    """Add `extra` rows to the training set, minus any whose template also appears in the test set
    (otherwise the scores would be inflated)."""
    if extra is None or not len(extra):
        return tr, 0
    test_keys = set(te.text.map(data_loader.dedup_key))
    extra = extra[~extra.text.map(data_loader.dedup_key).isin(test_keys)]
    return pd.concat([tr, extra[["text", "label", "sender"]]], ignore_index=True), len(extra)


def strata(df):
    """Split so each source keeps its smishing/legit ratio in both train and test."""
    return df.source.astype(str) + "|" + df.label.astype(str)


def hide_senders(tr):
    """Sender dropout: every training message that has a sender is added a second time without it.
    Otherwise "no sender" is learned as a sign of legit (most sender-less training rows are legit),
    and smishing in sms_taglish.csv, which has no senders, gets missed."""
    with_sender = tr[tr.sender.astype(str).str.strip() != ""]
    return pd.concat([tr, with_sender.assign(sender="")], ignore_index=True)


def train_and_score(tr, te, features="both"):
    """Fit the features and all models on `tr`. `features` is a key of FEATURE_SETS.
    Returns (vec, sc, models, smishing probabilities on `te`, feature matrix of `te`)."""
    tr = hide_senders(tr)
    vec, sc = TfidfVectorizer(ngram_range=(1, 2)), MaxAbsScaler()

    def build(d, fit_=False):
        parts = []
        if features in ("both", "words"):
            txt = d.text.apply(clean)
            parts.append(vec.fit_transform(txt) if fit_ else vec.transform(txt))
        if features in ("both", "hand"):
            H = np.array([hand(t, snd) for t, snd in zip(d.text, d.sender)])
            parts.append(csr_matrix(sc.fit_transform(H) if fit_ else sc.transform(H)))
        return hstack(parts).tocsr()

    Xtr, Xte = build(tr, True), build(te)
    models = make_models()
    probs = {n: m.fit(Xtr, tr.label).predict_proba(Xte)[:, 1] for n, m in models.items()}
    return vec, sc, models, probs, Xte


def metrics_table(y, probs, cutoff=0.5):
    """Metrics for each model when messages scoring >= `cutoff` are called smishing."""
    rows = []
    for name, p in probs.items():
        pred = (np.asarray(p) >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y, pred),
                     "Precision": precision_score(y, pred, zero_division=0),
                     "Recall": recall_score(y, pred, zero_division=0),
                     "F1": f1_score(y, pred, zero_division=0),
                     "Missed scams (FN)": int(fn),
                     "False alarms (FP)": int(fp)})
    return pd.DataFrame(rows).set_index("Model").round(3)


def fit(df, extra=None):
    """Train all models. `df` = the core Filipino/Taglish set, split 80/20. `extra` = training-only
    datasets and/or flagged messages, added to the TRAINING set only, so the test set is always
    held-out core data."""
    tr, te = train_test_split(df, test_size=0.2, stratify=strata(df), random_state=42)
    tr, n_extra = add_extra(tr, te, extra)
    vec, sc, models, probs, _ = train_and_score(tr, te)
    # A TF-IDF word can share a name with a hand-made feature (the word "length"); keep names unique
    names = [f"{w} (word)" if w in HAND_NAMES else w for w in vec.get_feature_names_out()] + HAND_NAMES

    return {"vec": vec, "sc": sc, "models": {n: models[n] for n in MODEL_NAMES},
            "y_test": te.label.to_numpy(), "test_probs": probs, "test_source": te.source.to_numpy(),
            "importance": model_importance(models, names), "feature_names": names,
            "test_keys": set(te.text.map(data_loader.dedup_key)),
            "n_train": len(tr), "n_test": len(te), "n_extra": n_extra, "df": df}


def model_importance(models, names):
    """Each model's own view of feature importance.
    LR and SVM: signed weights (+ pushes toward smishing, - toward legit; hand features are
    scaled to 0..1 like TF-IDF, so weights are comparable). DT: impurity importance (unsigned)."""
    svm = models["SVM"].calibrated_classifiers_[0].estimator
    svm_w = svm.coef_.toarray()[0] if hasattr(svm.coef_, "toarray") else np.asarray(svm.coef_)[0]
    return {"Logistic Regression": pd.Series(models["Logistic Regression"].coef_[0], index=names),
            "SVM": pd.Series(svm_w, index=names),
            "Decision Tree": pd.Series(models["Decision Tree"].feature_importances_, index=names)}


def permutation_drop(models, Xte, y, rng, n_shuffles=5):
    """For each hand-made feature: average drop in accuracy and recall when that feature's
    column is shuffled in the test fold (so the model can no longer use it)."""
    n_hand = len(HAND_NAMES)
    words, H = Xte[:, :-n_hand], Xte[:, -n_hand:].toarray()
    rows = []
    for name in MODEL_NAMES:
        m = models[name]
        base = m.predict(Xte)
        base_acc, base_rec = accuracy_score(y, base), recall_score(y, base, zero_division=0)
        for j, feat in enumerate(HAND_NAMES):
            acc, rec = [], []
            for _ in range(n_shuffles):
                Hp = H.copy()
                Hp[:, j] = rng.permutation(Hp[:, j])
                p = m.predict(hstack([words, csr_matrix(Hp)]).tocsr())
                acc.append(base_acc - accuracy_score(y, p))
                rec.append(base_rec - recall_score(y, p, zero_division=0))
            rows.append({"Model": name, "Feature": feat, "Accuracy drop": np.mean(acc), "Recall drop": np.mean(rec)})
    return pd.DataFrame(rows)


def mcnemar_p(y, pred_a, pred_b):
    """Exact McNemar test: do two models make significantly different mistakes on the same messages?"""
    a_right, b_right = pred_a == y, pred_b == y
    b, c = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def cross_validate(df, extra, n_repeats=5, n_splits=5, features="both"):
    """Repeated stratified k-fold on the core Filipino/Taglish set. Every message is tested once per
    repeat; `extra` is added to each training fold only. Returns out-of-fold probabilities per repeat
    (so metrics can be recomputed at any cutoff without retraining). The folds depend only on the
    data and the repeat number, so runs with different `features` are paired message by message.
    Permutation importance needs the hand-made features, so it is only run for features="both"."""
    y = df.label.to_numpy()
    runs, perm = [], []
    rng = np.random.default_rng(0)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        oof = {n: np.zeros(len(df)) for n in make_models()}
        fold = np.zeros(len(df), dtype=int)
        for k, (a, b) in enumerate(skf.split(df, strata(df))):
            tr, te = df.iloc[a], df.iloc[b]
            tr, _ = add_extra(tr, te, extra)
            _, _, models, probs, Xte = train_and_score(tr, te, features)
            for n, p in probs.items():
                oof[n][b] = p
            fold[b] = k
            if r == 0 and features == "both":  # permutation importance on the first repeat's folds
                perm.append(permutation_drop(models, Xte, te.label.to_numpy(), rng))
        runs.append({"oof": oof, "fold": fold})
    perm = pd.concat(perm).groupby(["Model", "Feature"], sort=False).mean().reset_index() if perm else None
    return {"y": y, "runs": runs, "n_splits": n_splits, "perm": perm}


def cv_summary(cv, cutoff=0.5):
    """Mean ± std of each metric over all folds of all repeats."""
    y, per_fold = cv["y"], []
    for run in cv["runs"]:
        for k in range(cv["n_splits"]):
            m = run["fold"] == k
            per_fold.append(metrics_table(y[m], {n: p[m] for n, p in run["oof"].items()}, cutoff))
    stacked = pd.concat(per_fold)
    mean, std = stacked.groupby(level=0, sort=False).mean(), stacked.groupby(level=0, sort=False).std()
    cols = ["Accuracy", "Precision", "Recall", "F1"]
    table = pd.DataFrame({c: [f"{mean.loc[i, c]:.3f} ± {std.loc[i, c]:.3f}" for i in mean.index] for c in cols},
                         index=mean.index)
    return table, mean[cols]


def cv_significance(cv, cutoff=0.5):
    """McNemar test for every pair of models, run on each repeat's out-of-fold predictions."""
    y, names = cv["y"], list(cv["runs"][0]["oof"])
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ps = [mcnemar_p(y, run["oof"][a] >= cutoff, run["oof"][b] >= cutoff) for run in cv["runs"]]
            acc_a = np.mean([((run["oof"][a] >= cutoff) == y).mean() for run in cv["runs"]])
            acc_b = np.mean([((run["oof"][b] >= cutoff) == y).mean() for run in cv["runs"]])
            med = float(np.median(ps))
            rows.append({"Model A": a, "Model B": b,
                         "Accuracy A": round(acc_a, 3), "Accuracy B": round(acc_b, 3),
                         "Median p-value": round(med, 4),
                         "Significant repeats": f"{sum(p < 0.05 for p in ps)} of {len(ps)}",
                         "Verdict (p < 0.05)": ("Significant: " + (a if acc_a > acc_b else b) + " is better")
                                               if med < 0.05 else "Not significant"})
    return pd.DataFrame(rows)


def feature_set_table(cvs, cutoff=0.5):
    """`cvs` = {feature set: cross_validate result}, all run on the same folds. For each model and
    feature set: mean CV recall and F1, and a McNemar test against "both" (median p over repeats)."""
    y, rows = cvs["both"]["y"], []
    for name in MODEL_NAMES:
        for fs, cv in cvs.items():
            _, mean = cv_summary(cv, cutoff)
            row = {"Model": name, "Features": FEATURE_SETS[fs],
                   "Recall": round(mean.loc[name, "Recall"], 3), "F1": round(mean.loc[name, "F1"], 3)}
            if fs != "both":
                ps = [mcnemar_p(y, ra["oof"][name] >= cutoff, rb["oof"][name] >= cutoff)
                      for ra, rb in zip(cvs["both"]["runs"], cv["runs"])]
                med = float(np.median(ps))
                worse = mean.loc[name, "F1"] < cv_summary(cvs["both"], cutoff)[1].loc[name, "F1"]
                row["p vs words + hand-made"] = round(med, 4)
                row["Verdict (p < 0.05)"] = (("Significantly worse" if worse else "Significantly better")
                                             if med < 0.05 else "Not significantly different")
            rows.append(row)
    return pd.DataFrame(rows).set_index(["Model", "Features"])


def recommend(cv, cutoff=0.5):
    """Pick the model to deploy from the CV results: highest recall (missing a scam is the costly
    error), F1 as tie-breaker. Also checks the lead is significant and suggests a cutoff."""
    _, mean = cv_summary(cv, cutoff)
    ranked = mean.drop(BASELINE).sort_values(["Recall", "F1"], ascending=False)
    best, runner_up = ranked.index[0], ranked.index[1]
    sig = cv_significance(cv, cutoff)
    pair = sig[((sig["Model A"] == best) & (sig["Model B"] == runner_up)) |
               ((sig["Model A"] == runner_up) & (sig["Model B"] == best))].iloc[0]
    # Suggested cutoff: maximise F2 (weights recall twice as much as precision) on the out-of-fold predictions
    y, grid = cv["y"], np.round(np.arange(0.05, 0.96, 0.05), 2)
    f2 = [np.mean([fbeta_score(y, run["oof"][best] >= c, beta=2, zero_division=0) for run in cv["runs"]])
          for c in grid]
    top = max(f2)
    best_cut = min((c for c, v in zip(grid, f2) if v >= top - 1e-9), key=lambda c: abs(c - 0.5))
    return {"best": best, "runner_up": runner_up, "mean": mean, "p": pair["Median p-value"],
            "cutoff": float(best_cut), "cut_summary": cv_summary(cv, float(best_cut))[1].loc[best]}


def load_base(csv_path=BASE_CSV):
    df = pd.read_csv(csv_path, dtype={"sender": str}).dropna(subset=["text", "label"])
    df["label"] = df.label.astype(int)
    df["sender"] = df["sender"].fillna("") if "sender" in df else ""
    df["source"] = Path(csv_path).name
    return df[["text", "label", "sender", "source"]]

def load_data(data_dir=data_loader.DATA_DIR, csv_path=BASE_CSV):
    """Returns (core, train_extra, summary, n_conflicts). core = sms_taglish.csv + files with role
    "core", one row per template; it is what gets split and cross-validated. train_extra = files
    with role "train", used for training only. summary = the per-file report from data_loader."""
    extra, summary = data_loader.load_extra(data_dir)
    core, n_conf = data_loader.dedupe(pd.concat([load_base(csv_path), extra[extra.role == "core"]],
                                                ignore_index=True))
    train_extra, n_conf_tr = data_loader.dedupe(extra[extra.role == "train"])
    return core.drop(columns="role", errors="ignore"), train_extra, summary, n_conf + n_conf_tr


def predict_all(art, msg, sender=""):
    X = hstack([art["vec"].transform([clean(msg)]),
                csr_matrix(art["sc"].transform([hand(msg, sender)]))]).tocsr()
    return {n: float(m.predict_proba(X)[0][1]) for n, m in art["models"].items()}

```
`pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    slow: runs the whole Streamlit app (skip with -m "not slow")

```
`requirements-dev.txt`:

```txt
-r requirements.txt
pytest==9.1.1

```
`requirements.txt`:

```txt
streamlit==1.65.0
pandas==3.0.6
numpy==2.5.4
scipy==1.18.1
scikit-learn==1.9.1
matplotlib==3.11.2
openpyxl==3.1.5

```
`sms_taglish.csv`:

```csv
CSV Schema (1 sample row):
Headers: sender, text, label
Sample: "", "Congratulations winner! Na-select ka sa Coca-Cola Promo, P75,000 ang premyo mo. Text back your full name agad.", "1"
... [198 more rows omitted]

```
`tests\test_features.py`:

```py
"""Unit tests for the text features and the datasets/ loader. Run with: pytest"""
import json

import pandas as pd
import pytest

import data_loader
from features import HAND_NAMES, clean, hand, sender_type, urls


def feat(text, sender=""):
    return dict(zip(HAND_NAMES, hand(text, sender)))


# ---------- sender_type ----------
@pytest.mark.parametrize("sender, expected", [
    ("", "unknown"),
    (None, "unknown"),
    ("nan", "unknown"),
    ("09171234567", "number"),
    ("+63 917 123 4567", "number"),
    ("BDO", "alpha"),
    ("GCash", "alpha"),
    ("2958", "alpha"),  # 4-digit short codes are official senders, like named IDs
    ("+63960****655", "number"),  # masked numbers in public datasets
    ("+6396***32373", "number"),
    ("+63981****875", "number"),  # prefix newer than the 2023 network prefix list
    ("789BIngo 5", "alpha"),
    ("****", "alpha"),
])
def test_sender_type(sender, expected):
    assert sender_type(sender) == expected


def test_sender_features_are_one_hot():
    for sender in ["", "09171234567", "BDO"]:
        f = feat("hello", sender)
        assert f["sender_is_number"] + f["sender_is_alpha"] + f["sender_unknown"] == 1


# ---------- has_phone ----------
@pytest.mark.parametrize("text, expected", [
    ("Send P300 sa 09561234567", 1),
    ("call +639171234567 now", 1),
    ("tawag 0917-123-4567", 1),
    ("tawag 0917 123 4567", 1),
    ("OTP 482913", 0),
    ("acct 1234567890123", 0),
])
def test_has_phone(text, expected):
    assert feat(text)["has_phone"] == expected


# ---------- URLs ----------
@pytest.mark.parametrize("text, expected", [
    ("I-verify agad dito: bit.ly/gc-verify", ["bit.ly/gc-verify"]),
    ("Go to gcash-verify.top/login now", ["gcash-verify.top/login"]),
    ("Official: help.gcash.com.", ["help.gcash.com"]),
    ("Visit https://bdo.com.ph/help.", ["https://bdo.com.ph/help"]),
    ("Telegram: t.me/easyearnph", ["t.me/easyearnph"]),
    ("Bayaran P500.00 ngayon", []),
    ("Your bill is P1,699.50", []),
    ("Ref: SM2210. Thanks", []),
])
def test_urls(text, expected):
    assert urls(text) == expected


@pytest.mark.parametrize("text, has_url, short_url, suspicious_tld", [
    ("I-verify agad: bit.ly/gc-verify", 1, 1, 0),
    ("Claim now: shopee-ph-prize.xyz", 1, 0, 1),
    ("Restore: gcash-help.cc/restore", 1, 0, 1),
    ("Official updates at help.gcash.com.", 1, 0, 0),
    ("Bisitahin ang globe.com.ph", 1, 0, 0),
    ("Your BDO OTP is 482913.", 0, 0, 0),
])
def test_url_features(text, has_url, short_url, suspicious_tld):
    f = feat(text)
    assert (f["has_url"], f["short_url"], f["suspicious_tld"]) == (has_url, short_url, suspicious_tld)


def test_clean_keeps_words_inside_links_and_masks_numbers():
    out = clean("Go to gcash-verify.top/login, P500 now")
    assert out.split()[:6] == ["go", "to", "URL", "gcash", "verify", "top"]
    assert "NUM" in out and "500" not in out


def test_keyword_counts():
    f = feat("URGENT: claim now ang premyo mo sa GCash agad")
    assert f["urgency_count"] >= 3  # urgent, claim now, agad
    assert f["money_count"] >= 2    # premyo, gcash


def test_hand_feature_count_matches_names():
    assert len(hand("anything")) == len(HAND_NAMES)


# ---------- datasets/ loader ----------
def test_loader_handles_kaggle_formats(tmp_path):
    # UCI-style: v1/v2 columns, latin-1, junk extra column
    pd.DataFrame({"v1": ["ham", "spam"], "v2": ["See you later café", "WINNER! claim £900"],
                  "Unnamed: 2": [None, None]}).to_csv(tmp_path / "uci.csv", index=False, encoding="latin-1")
    # Numeric labels written as floats, with a sender column
    pd.DataFrame({"message": ["a scam", "a legit"], "target": [1.0, 0.0],
                  "from": ["09171234567", "BDO"]}).to_csv(tmp_path / "num.csv", index=False)
    # 3 classes; config keeps only smishing vs ham
    pd.DataFrame({"LABEL": ["ham", "spam", "smishing"], "TEXT": ["hi", "buy now", "BDO verify"]}
                 ).to_csv(tmp_path / "three.csv", index=False)
    # Unusable file
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(tmp_path / "bad.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(
        {"three.csv": {"positive": ["smishing"], "negative": ["ham"]}}))

    extra, summary = data_loader.load_extra(tmp_path)
    status = summary.set_index("File").Status
    assert status["bad.csv"].startswith("Skipped")
    assert (status.drop("bad.csv") == "OK").all()

    assert set(extra.columns) >= {"text", "label", "sender", "source"}
    rows = extra.set_index("text")
    assert rows.loc["WINNER! claim £900", "label"] == 1
    assert rows.loc["See you later café", "label"] == 0
    assert rows.loc["a scam", "label"] == 1 and rows.loc["a scam", "sender"] == "09171234567"
    assert rows.loc["BDO verify", "label"] == 1
    assert "buy now" not in rows.index  # 'spam' dropped by config


def test_loader_reads_xlsx_roles_and_fixes_mojibake(tmp_path):
    pd.DataFrame({"text": ["Weâ€™re here", "Ok"], "category": ["spam", "notifs"],
                  "sender": ["+63960****655", "GCash"]}).to_excel(tmp_path / "ph.xlsx", index=False)
    # PHONE is a yes/no flag here, not a sender column
    pd.DataFrame({"LABEL": ["ham"], "TEXT": ["hello"], "PHONE": ["No"]}).to_csv(tmp_path / "en.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(
        {"ph.xlsx": {"role": "core", "positive": ["spam"], "negative": ["notifs"]}}))

    extra, summary = data_loader.load_extra(tmp_path)
    assert (summary.Status == "OK").all()
    rows = extra.set_index("text")
    assert rows.loc["We’re here", "label"] == 1 and rows.loc["We’re here", "role"] == "core"
    assert rows.loc["We’re here", "sender"] == "+63960****655"
    assert rows.loc["hello", "role"] == "train" and rows.loc["hello", "sender"] == ""


def test_loader_rejects_unknown_role(tmp_path):
    pd.DataFrame({"text": ["x"], "label": [1]}).to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"a.csv": {"role": "test"}}))
    assert data_loader.load_extra(tmp_path)[1].Status[0].startswith("Skipped: role")


def test_fix_mojibake_leaves_clean_text_alone():
    assert data_loader.fix_mojibake("Dasmariñas ₱500 café") == "Dasmariñas ₱500 café"
    assert data_loader.fix_mojibake("Shiftâ€™s Over") == "Shift’s Over"


def test_dedupe_by_template_and_drops_conflicts():
    df = pd.DataFrame({"text": ["Your OTP is 111111", "Your OTP is 222222",           # same template
                                "Claim at a.xyz", "Claim at b.xyz",                   # same template...
                                "hello"],
                       "label": [0, 0, 1, 0, 0]})                                     # ...labelled both ways
    out, n_conflicts = data_loader.dedupe(df)
    assert list(out.text) == ["Your OTP is 111111", "hello"]
    assert n_conflicts == 2


def test_loader_skip_and_signature(tmp_path):
    assert data_loader.load_extra(tmp_path)[1].empty
    before = data_loader.signature(tmp_path)
    pd.DataFrame({"text": ["x"], "label": [1]}).to_csv(tmp_path / "a.csv", index=False)
    assert data_loader.signature(tmp_path) != before  # adding a file changes the cache key
    (tmp_path / "config.json").write_text(json.dumps({"a.csv": {"skip": True}}))
    extra, summary = data_loader.load_extra(tmp_path)
    assert extra.empty and summary.Status[0].startswith("Skipped")

```
`tests\test_model.py`:

```py
"""Tests for the training and evaluation code in model.py, plus a smoke test of the Streamlit app.
Run with: pytest"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_loader
import model

SCAMS = ["GCash: Na-lock ang account mo. I-verify agad: gcash-verify{}.top",
         "Congrats! Nanalo ka ng P{}000. I-claim now: promo{}.xyz",
         "BDO: suspicious login. Update now at bdo-secure{}.cc"]
LEGIT = ["Hi Ma, nasa jeep na ako, {} mins pa", "Your OTP is {}1234. Do not share it",
         "Salamat sa order #{}! Arriving in 2 days"]


def toy(n=12, source="toy.csv"):
    rows = [(SCAMS[i % 3].format(i, i), 1, "+63960****65" + str(i % 10)) for i in range(n)]
    rows += [(LEGIT[i % 3].format(i * 7), 0, "GCash") for i in range(n)]
    # Make every text a different template (dedup_key masks digits and links)
    rows = [(f"{t} {'abcdefghijklmnopqrstuvwxyz'[i % 26] * (i // 26 + 1)}", y, s) for i, (t, y, s) in enumerate(rows)]
    return pd.DataFrame(rows, columns=["text", "label", "sender"]).assign(source=source)


def test_hide_senders_adds_a_senderless_copy_of_rows_with_a_sender():
    tr = pd.DataFrame({"text": ["a", "b"], "label": [1, 0], "sender": ["09171234567", ""]})
    out = model.hide_senders(tr)
    assert len(out) == 3
    assert list(out.sender) == ["09171234567", "", ""]
    assert list(out.text) == ["a", "b", "a"]


def test_add_extra_drops_templates_in_the_test_set():
    te = pd.DataFrame({"text": ["Your OTP is 111111"], "label": [0], "sender": [""]})
    tr = pd.DataFrame({"text": ["hello"], "label": [0], "sender": [""]})
    extra = pd.DataFrame({"text": ["Your OTP is 999999", "new message"], "label": [0, 1], "sender": ["", ""]})
    out, n = model.add_extra(tr, te, extra)
    assert n == 1 and "Your OTP is 999999" not in set(out.text)


def test_fit_keeps_training_only_rows_out_of_the_test_set():
    core = toy()
    extra = pd.DataFrame({"text": ["English smishing: verify at paypal-x.top"], "label": [1], "sender": [""]})
    art = model.fit(core, extra)
    assert art["n_test"] == round(len(core) * 0.2) and art["n_extra"] == 1
    assert data_loader.dedup_key(extra.text[0]) not in art["test_keys"]
    probs = model.predict_all(art, "GCash: I-verify agad: gcash-login.top", "+63960****111")
    assert set(probs) == set(model.MODEL_NAMES) and all(0 <= p <= 1 for p in probs.values())


def test_cross_validation_folds_are_paired_across_feature_sets():
    core = toy()
    cvs = {fs: model.cross_validate(core, None, n_repeats=2, features=fs) for fs in model.FEATURE_SETS}
    for cv in cvs.values():  # same messages in the same folds, so McNemar pairs are valid
        assert all((r["fold"] == b["fold"]).all() for r, b in zip(cv["runs"], cvs["both"]["runs"]))
    assert cvs["both"]["perm"] is not None and cvs["words"]["perm"] is None
    table = model.feature_set_table(cvs)
    assert len(table) == len(model.MODEL_NAMES) * len(model.FEATURE_SETS)
    assert pd.isna(table.loc[("SVM", model.FEATURE_SETS["both"]), "p vs words + hand-made"])  # the reference row


def test_mcnemar_identical_predictions_is_not_significant():
    y = np.array([1, 0, 1, 0])
    assert model.mcnemar_p(y, y == 1, y == 1) == 1.0


def test_load_data_puts_core_files_in_the_split_and_train_files_aside(tmp_path):
    data = tmp_path / "datasets"
    data.mkdir()
    toy(6, "x").drop(columns="source").to_csv(data / "core.csv", index=False)
    pd.DataFrame({"text": ["english one", "english two"], "label": [1, 0]}).to_csv(data / "en.csv", index=False)
    (data / "config.json").write_text('{"core.csv": {"role": "core"}}')
    base = tmp_path / "base.csv"
    pd.DataFrame({"sender": [""], "text": ["our own message"], "label": [0]}).to_csv(base, index=False)
    core, train_extra, summary, n_conf = model.load_data(data, base)
    assert set(core.source) == {"base.csv", "core.csv"} and len(core) == 13
    assert list(train_extra.text) == ["english one", "english two"]
    assert n_conf == 0 and (summary.Status == "OK").all()


@pytest.mark.slow
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=600).run()
    assert not at.exception and not at.error

```