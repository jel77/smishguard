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
