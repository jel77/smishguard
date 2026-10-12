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

Requires **Python 3.11+** (tested on 3.14). Windows commands:

```powershell
git clone <this repo>
cd smishguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. On macOS/Linux, activate with `source .venv/bin/activate`.

To run the tests:

```powershell
pip install -r requirements-dev.txt
pytest
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
  Logistic Regression · Decision Tree · SVM   (+ random-guess baseline)
         │
         ▼
  smishing probability → SMISHING if ≥ cutoff (default 0.50)
```

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
| 📊 **Model comparison** | Held-out test results, a recall vs false-alarm chart across cutoffs, **cross-validation** with **McNemar's significance test**, and a **recommendation** for which model to deploy. |
| 🧠 **Top features** | Hand-made features ranked across all 3 models, permutation importance from cross-validation, and each model's top features (including the Decision Tree's rules). |
| 🗂️ **Dataset** | The core dataset, plus a per-file report for anything in `datasets/`. |

The **Smishing cutoff** slider in the sidebar sets the probability at which a message counts as
smishing. Lower it to catch more scams (higher recall) at the cost of more false alarms.

> Flagged messages are kept in memory, are shared by everyone using the same running app, and are
> lost when it restarts. To keep them, download `flagged.csv` and add the rows to `sms_taglish.csv`.

---

## Data

### Core dataset: `sms_taglish.csv`

The Filipino/Taglish dataset. It is the **only** source of test messages, so every score measures
Taglish smishing.

| Column | Values |
|---|---|
| `sender` | Optional. A mobile number (`09171234567`), a sender ID (`BDO`, `GCash`) or blank |
| `text` | The SMS message |
| `label` | `1` = smishing, `0` = legit |

> **To do:** the `sender` column is currently blank for all rows, so the three sender features
> have no effect yet. Filling it in lets the models use them.

### Extra training data: `datasets/`

Drop CSVs (e.g. from Kaggle) into [datasets/](datasets/) and refresh the app. They are used for
**training only**. Text and label columns are detected automatically (`v1`/`v2`, `text`/`label`,
`message`/`category`, …), `spam`/`ham` labels are converted to 1/0, and any row that also appears
in the test set is removed. If a file is read wrong, fix it with `datasets/config.json`.
See [datasets/README.md](datasets/README.md) for details.

---

## Current results

*As of the 199-message `sms_taglish.csv` (104 smishing, 95 legit), no extra datasets, cutoff 0.50.
Stratified 5-fold cross-validation × 5 repeats (mean ± std over 25 folds). These numbers change as
the data grows; the app always shows the current ones.*

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.906 ± 0.040 | 0.968 ± 0.037 | 0.848 ± 0.063 | 0.903 ± 0.042 |
| Decision Tree | 0.907 ± 0.046 | 0.917 ± 0.060 | 0.908 ± 0.062 | 0.911 ± 0.044 |
| **SVM** | **0.967 ± 0.035** | 0.967 ± 0.042 | **0.971 ± 0.046** | **0.969 ± 0.034** |
| Random guess (baseline) | 0.504 ± 0.068 | 0.524 ± 0.061 | 0.578 ± 0.063 | 0.549 ± 0.062 |

- **All three models beat random guessing** (McNemar p < 0.001).
- **SVM is significantly better than LR (p = 0.001) and DT (p = 0.004).** LR and DT are not
  significantly different (p = 1.0).
- **Lowering the cutoff raises recall and lowers precision:** at 0.30, LR recall goes from 0.848 to
  0.956 and SVM recall from 0.971 to 0.990. At that cutoff, SVM's lead over LR is no longer significant.
- **Strongest features:** `has_phone` ranks #1 for LR and SVM and #2 for DT. `suspicious_tld` is #2
  for LR and SVM. `urgency_count`, `money_count` and `short_url` rank between #3 and #6 for both.
  The Decision Tree mostly splits on words instead (e.g. `url help`, as in official links like
  help.gcash.com, points to legit).

---

## Project structure

```
app.py                  Streamlit app: training, evaluation and the UI
features.py             Text cleaning and the 11 hand-made features
data_loader.py          Loads and standardises CSVs from datasets/
sms_taglish.csv         Core Filipino/Taglish dataset (training + test)
datasets/               Extra training CSVs (README.md, config.example.json)
tests/test_features.py  Unit tests for features and the loader
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
