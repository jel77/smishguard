# datasets/

Put extra training CSVs here (for example from Kaggle). Every `.csv` in this folder is
loaded automatically when the app starts or the page is refreshed.

- These files are used for **training only**. The test set is always the 20% held-out
  part of `../sms_taglish.csv`, so the scores still measure Filipino/Taglish smishing.
- Rows whose text also appears in the test set are dropped automatically.
- Duplicate messages across files are removed.
- The **Dataset** tab in the app lists each file, how many rows were used, and why a
  file was skipped.

## Accepted format

Each CSV needs one text column and one label column. Common names are detected for you:

| | Detected column names |
|---|---|
| Text | `text`, `message`, `sms`, `msg`, `v2`, `content`, `body`, `sms_text`, `message_text` |
| Label | `label`, `v1`, `category`, `class`, `type`, `target`, `spam`, `is_spam`, `labels` |
| Sender (optional) | `sender`, `from`, `sender_id`, `address`, `phone`, `number` |

Label values are converted to `1` (smishing/scam) or `0` (legit):

- `1`: `1`, `spam`, `smishing`, `smish`, `scam`, `phishing`, `fraud`, `true`, `yes`
- `0`: `0`, `ham`, `legit`, `legitimate`, `normal`, `not spam`, `false`, `no`

Rows with any other label are dropped (shown in the "Dropped" column).

The sender column is optional. A sender that is a plain mobile number (`09171234567`, `+639...`)
counts as `number`; a name or short code (`BDO`, `GCash`, `2958`) counts as `alpha`; blank
counts as `unknown`. Files without a sender column are treated as all `unknown`.

## config.json (only if auto-detection gets a file wrong)

Create `config.json` in this folder, keyed by file name. Every field is optional.
See `config.example.json`.

```json
{
  "kaggle_smishing.csv": {
    "text": "TEXT",
    "label": "LABEL",
    "positive": ["smishing"],
    "negative": ["ham"]
  },
  "big_english_spam.csv": { "skip": true }
}
```

- `text` / `label` / `sender`: the exact column names in that file (`sender` is optional).
- `positive`: the label values that count as smishing (`1`).
- `negative`: the label values that count as legit (`0`). If you give only `positive`,
  every other value counts as legit.
- `skip`: `true` ignores the file without deleting it.

Tip: some Kaggle sets have three classes, like `ham` / `spam` / `smishing`. Our paper only
targets smishing, so to keep general spam out, use `"positive": ["smishing"], "negative": ["ham"]`
(rows labelled `spam` are then dropped).
