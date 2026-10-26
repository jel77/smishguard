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
