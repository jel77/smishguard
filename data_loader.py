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
