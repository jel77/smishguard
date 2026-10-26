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
