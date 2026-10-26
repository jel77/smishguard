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
