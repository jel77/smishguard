"""Loads the extra training CSVs in datasets/ (e.g. from Kaggle) into one text/label table.

Each CSV is mapped to two columns: `text` and `label` (1 = smishing/spam, 0 = legit).
Columns are auto-detected from common names. If that guesses wrong, add an entry for the
file in datasets/config.json (see datasets/README.md).
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "datasets"

TEXT_COLS = ["text", "message", "sms", "msg", "v2", "content", "body", "sms_text", "message_text"]
SENDER_COLS = ["sender", "from", "sender_id", "address", "phone", "number"]
LABEL_COLS = ["label", "v1", "category", "class", "type", "target", "spam", "is_spam", "labels"]
POSITIVE = {"1", "spam", "smishing", "smish", "scam", "phishing", "fraud", "true", "yes"}
NEGATIVE = {"0", "ham", "legit", "legitimate", "normal", "not spam", "false", "no"}


def _read_csv(path):
    # Many Kaggle SMS files (e.g. UCI spam.csv) are latin-1, not UTF-8
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode file")


def _find(cols, candidates):
    lower = {c.strip().lower(): c for c in cols}
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


def load_config(data_dir=DATA_DIR):
    config_file = Path(data_dir) / "config.json"
    if config_file.exists():
        return json.loads(config_file.read_text(encoding="utf-8"))
    return {}


def signature(data_dir=DATA_DIR):
    """Changes whenever a CSV or the config is added, removed or edited (used as a cache key)."""
    data_dir, config_file = Path(data_dir), Path(data_dir) / "config.json"
    if not data_dir.exists():
        return ()
    files = sorted(data_dir.glob("*.csv")) + ([config_file] if config_file.exists() else [])
    return tuple((f.name, f.stat().st_size, f.stat().st_mtime) for f in files)


def load_extra(data_dir=DATA_DIR):
    """Returns (combined DataFrame[text, label, sender, source], per-file summary DataFrame)."""
    data_dir = Path(data_dir)
    config = load_config(data_dir)
    frames, summary = [], []
    for path in sorted(data_dir.glob("*.csv")) if data_dir.exists() else []:
        cfg = config.get(path.name, {})
        row = {"File": path.name, "Text column": None, "Label column": None, "Sender column": None,
               "Rows used": 0, "Smishing": 0, "Legit": 0, "Dropped": 0, "Status": "OK"}
        try:
            if cfg.get("skip"):
                raise ValueError("skipped in config.json")
            df = _read_csv(path)
            tcol = cfg.get("text") or _find(df.columns, TEXT_COLS)
            lcol = cfg.get("label") or _find(df.columns, LABEL_COLS)
            if tcol not in df.columns or lcol not in df.columns:
                raise ValueError(f"can't find text/label columns in {list(df.columns)[:8]}; "
                                 "set them in config.json")
            scol = cfg.get("sender") or _find(df.columns, SENDER_COLS)
            row["Text column"], row["Label column"], row["Sender column"] = tcol, lcol, scol
            out = pd.DataFrame({"text": df[tcol].astype(str).str.strip(),
                                "label": _to_label(df[lcol], cfg.get("positive"), cfg.get("negative")),
                                "sender": df[scol].fillna("").astype(str).str.strip() if scol in df else ""})
            before = len(out)
            out = out[(out.text != "") & (out.text.str.lower() != "nan")].dropna(subset=["label"])
            out["label"] = out.label.astype(int)
            out["source"] = path.name
            row.update({"Rows used": len(out), "Smishing": int(out.label.sum()),
                        "Legit": int((out.label == 0).sum()), "Dropped": before - len(out)})
            if out.empty:
                raise ValueError("no usable rows (check the label values in config.json)")
            frames.append(out)
        except Exception as e:  # one bad file shouldn't break the app
            row["Status"] = f"Skipped: {e}"
        summary.append(row)

    extra = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame({"text": [], "label": [], "sender": [], "source": []}))
    extra = extra.drop_duplicates(subset="text").reset_index(drop=True)
    return extra, pd.DataFrame(summary)
