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
    st.write("Cross-validates every model again with only the TF-IDF words, and with only the 12 hand-made "
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
