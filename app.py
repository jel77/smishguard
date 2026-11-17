import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import export_text
from sklearn.metrics import precision_score, recall_score

import data_loader
import model
from features import HAND_NAMES, sender_type
from model import (BASELINE, FEATURE_SETS, MODEL_NAMES, cv_significance, cv_summary, feature_set_table,
                   fit, metrics_table, predict_all, recommend)

st.set_page_config(page_title="SmishGuard", page_icon=":material/shield:", layout="wide")
# Colours, fonts and chart palettes live in .streamlit/config.toml (light and dark variants)

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


def section(title, text=None):
    """Section heading with an optional one-line description."""
    st.subheader(title, anchor=False)
    if text:
        st.caption(text)


# ---------- Header ----------
st.title("SmishGuard", anchor=False, icon=":material/shield:")
st.caption("Detecting banking, e-wallet and online payment smishing in Filipino and Taglish text messages.")

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
retrained = "art_retrained" in st.session_state
flags = flag_store()

with st.sidebar:
    st.markdown("### Detection settings")
    # Rounded so 0.05 steps compare exactly against the cutoff grid used in the charts
    cutoff = round(st.slider(
        "Smishing cutoff", 0.05, 0.95, 0.50, 0.05,
        help="A message is called SMISHING when a model's smishing probability is at or above this value. "
             "Lower it to catch more scams (higher recall) at the cost of more false alarms."), 2)
    st.caption("Applies to the Analyze and Model evaluation tabs.")
    st.markdown("### Model status")
    if retrained:
        st.badge("Retrained with flags", color="orange", icon=":material/model_training:")
    else:
        st.badge("Base model", color="primary", icon=":material/verified:")
    st.caption(f"Trained on {art['n_train']:,} messages. Evaluated on {art['n_test']:,} held-out "
               f"Filipino/Taglish messages.")
    st.markdown(f"**Flagged messages:** {len(flags)}")
    st.caption("Models: Logistic Regression, Decision Tree and linear SVM on TF-IDF words plus "
               f"{len(HAND_NAMES)} hand-made features.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Core messages", f"{len(base_art['df']):,}", border=True,
          help="Filipino/Taglish messages that are split and cross-validated")
m2.metric("Training rows", f"{art['n_train']:,}", border=True,
          help="Includes training-only datasets and any flagged messages used for retraining")
m3.metric("Held-out test set", f"{art['n_test']:,}", border=True)
m4.metric("Smishing cutoff", f"{cutoff:.2f}", border=True)

tab_demo, tab_fb, tab_models, tab_feat, tab_data = st.tabs(
    ["Analyze", "Feedback", "Model evaluation", "Feature analysis", "Dataset"])

# (sender, message)
EXAMPLES = {
    "GCash scare + short link": ("+639561234567", "GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify"),
    "Prize scam": ("09281234567", "Congrats! Nanalo ka ng P30,000. I-claim now: promo-ph.xyz"),
    "Real bank OTP": ("BDO", "Your BDO OTP is 482913. Do not share this code with anyone."),
    "Casual Taglish chat": ("", "pre nasend ko na yung pambayad, check mo na lang"),
}

# ---------- Analyze ----------
with tab_demo:
    if retrained:
        st.info("Using the **retrained** model, which includes your flagged messages.",
                icon=":material/model_training:")
    left, right = st.columns([5, 7], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Message to analyze**")
            choice = st.selectbox("Example", ["Custom message"] + list(EXAMPLES),
                                  help="Load a sample message, or keep Custom message and write your own.")
            default_sender, default = EXAMPLES.get(choice, ("", ""))
            sender = st.text_input("Sender (optional)", value=default_sender, key=f"sender_{choice}",
                                   placeholder="e.g. 09171234567 or BDO")
            msg = st.text_area("SMS text", value=default, height=140, key=f"msg_{choice}",
                               placeholder="Paste the text message here")
            if st.button("Analyze message", type="primary", icon=":material/search:", width="stretch") \
                    and msg.strip():
                st.session_state["last"] = {"msg": msg.strip(), "sender": sender.strip(),
                                            "probs": predict_all(art, msg, sender)}

    with right:
        last = st.session_state.get("last")
        if not last:
            with st.container(border=True):
                st.markdown("**Result**")
                st.caption("Enter a message and select **Analyze message**. Each of the three models gives a "
                           "smishing probability, and the verdict below combines their votes.")
        else:
            probs = last["probs"]
            votes = sum(p >= cutoff for p in probs.values())
            n_models = len(probs)
            detail = f"{votes} of {n_models} models flag this message as smishing at cutoff {cutoff:.2f}."
            if votes == n_models:
                st.error(detail, title="Likely smishing", icon=":material/gpp_bad:")
            elif votes == 0:
                st.success(detail, title="Likely legitimate", icon=":material/verified_user:")
            else:
                st.warning(detail, title="Uncertain: the models disagree", icon=":material/help:")

            cols = st.columns(n_models)
            for col, (name, p) in zip(cols, probs.items()):
                with col.container(border=True):
                    st.markdown(f"**{name}**")
                    # Don't round 99.6% up to a certain-looking "100%" (or 0.4% down to "0%")
                    st.metric("Smishing probability", ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p:.0%}")
                    st.progress(min(max(p, 0.0), 1.0))
                    if p >= cutoff:
                        st.badge("Smishing", color="red")
                    else:
                        st.badge("Legitimate", color="green")

            snd = last["sender"]
            st.caption(f"Analyzed message. Sender: {snd or 'not given'} ({sender_type(snd)})")
            st.code(last["msg"], language=None, wrap_lines=True)

            st.space("small")
            st.markdown("**Is this result wrong?** Record the correct label to use it for retraining.")
            b1, b2 = st.columns(2)
            if b1.button("Mark as smishing", icon=":material/flag:", width="stretch"):
                flags[(last["msg"], last["sender"])] = 1
                st.toast("Recorded as smishing. See the Feedback tab.")
            if b2.button("Mark as legitimate", icon=":material/check_circle:", width="stretch"):
                flags[(last["msg"], last["sender"])] = 0
                st.toast("Recorded as legitimate. See the Feedback tab.")

# ---------- Feedback ----------
with tab_fb:
    section("Flagged messages", "Messages marked as misclassified in the Analyze tab. Retrain to teach the "
                                "models from them.")
    if not flags:
        st.info("Nothing flagged yet. Analyze a message, then use **Mark as smishing** or **Mark as legitimate**.",
                icon=":material/info:")
    else:
        fdf = pd.DataFrame({"text": [m for m, _ in flags], "label": list(flags.values()),
                            "sender": [snd for _, snd in flags]})
        in_test = fdf.text.map(data_loader.dedup_key).isin(base_art["test_keys"])
        shown = fdf.assign(label=fdf.label.map({1: "Smishing", 0: "Legit"}),
                           **{"In test set": in_test.map({True: "Yes (not used)", False: ""})})
        st.dataframe(shown, width="stretch", hide_index=True, alt="Flagged messages",
                     column_config={"text": "Message", "label": "Corrected label", "sender": "Sender"})
        if in_test.any():
            st.warning(f"{int(in_test.sum())} flagged message(s) are in the held-out test set, so retraining "
                       "leaves them out on purpose: training on test messages would make the scores look better "
                       "than they really are. Rephrase the message slightly if you want the model to learn from it.")

        repeat = st.slider("Weight of each flagged message (copies added to training)", 1, 20, 5,
                           help="One extra message barely moves a model trained on thousands of rows, so "
                                "repeating it gives it more weight.")
        c1, c2, c3 = st.columns(3)
        if c1.button("Retrain with flagged messages", type="primary", icon=":material/model_training:",
                     width="stretch"):
            core, train_extra, _, _ = load_data(sig)
            extra = pd.concat([train_extra] + [fdf] * repeat, ignore_index=True)
            with st.spinner("Retraining..."):
                st.session_state["art_retrained"] = fit(core, extra)
                st.session_state["retrained_sig"] = sig
            st.toast("Retrained. The Analyze tab now uses the updated models.")
            st.rerun()
        c2.download_button("Download flagged.csv", fdf.to_csv(index=False), file_name="flagged.csv",
                           mime="text/csv", icon=":material/download:", width="stretch")
        if c3.button("Clear flags", icon=":material/delete:", width="stretch"):
            flags.clear()
            st.rerun()

    if retrained:
        if st.button("Revert to the original model", icon=":material/undo:"):
            del st.session_state["art_retrained"]
            st.rerun()

    st.caption("Flags are shared by everyone using the app and are lost when the app restarts. To keep them, "
               "download flagged.csv and add its rows to sms_taglish.csv in GitHub.")

# ---------- Model evaluation ----------
with tab_models:
    # --- Single held-out split ---
    section("Held-out test set",
            f"Train: {art['n_train']:,} messages (incl. {art['n_extra']:,} training-only rows from datasets/ "
            f"and flags). Test: {art['n_test']:,} held-out core Filipino/Taglish messages. Cutoff {cutoff:.2f}.")
    results = metrics_table(art["y_test"], art["test_probs"], cutoff)
    st.dataframe(results, width="stretch", alt="Test set metrics per model")
    st.bar_chart(results[["Accuracy", "Precision", "Recall", "F1"]], horizontal=True, stack=False, sort=False,
                 height=380, alt="Bar chart of accuracy, precision, recall and F1 for each model on the test set")
    st.caption("Missed scams (FN) are the costly error here: a smishing text that got through. "
               f"With {art['n_test']} test messages, one message moves accuracy by "
               f"{1 / art['n_test']:.1%}; use the cross-validation below for the paper.")

    st.markdown("**Test results by source** (the same test set, split by the file each message came from)")
    src = art["test_source"]
    by_src = pd.concat({s: metrics_table(art["y_test"][src == s],
                                         {n: p[src == s] for n, p in art["test_probs"].items()}, cutoff)
                        for s in pd.unique(src)}, names=["Source", "Model"])
    st.dataframe(by_src.drop(index=BASELINE, level="Model"), width="stretch", alt="Test set metrics per source file")
    st.caption("sms_taglish.csv has no senders, while most messages in the other core files do. A big gap in "
               "recall between sources can mean the models lean on the sender features.")

    # --- Cutoff trade-off (H4) ---
    section("Recall vs false alarms by cutoff")
    pick = st.selectbox("Model", MODEL_NAMES, index=2, key="curve_model")
    y, p = art["y_test"], art["test_probs"][pick]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    curve = pd.DataFrame([{"Cutoff": c,
                           "Recall": recall_score(y, p >= c, zero_division=0),
                           "Precision": precision_score(y, p >= c, zero_division=0),
                           "False alarm rate": ((p >= c) & (y == 0)).sum() / max((y == 0).sum(), 1)}
                          for c in grid]).set_index("Cutoff")
    st.line_chart(curve, height=320, alt=f"Line chart of recall, precision and false alarm rate by cutoff for {pick}")
    st.caption(f"Lower cutoffs catch more scams (recall up) but flag more legit texts (false alarm rate up). "
               f"Current cutoff {cutoff:.2f}: recall {curve.Recall.asof(cutoff):.3f}, "
               f"false alarm rate {curve['False alarm rate'].asof(cutoff):.3f}.")

    # --- Cross-validation (H1, H2) ---
    section("Cross-validation",
            "Stratified 5-fold cross-validation on the Taglish data, repeated with different shuffles. Every "
            "message is tested once per repeat; datasets/ files are added to the training folds only.")
    n_rep = st.select_slider("Repeats", [1, 3, 5, 10], value=5, key="cv_repeats")
    if st.button("Run cross-validation", icon=":material/play_arrow:") or st.session_state.get("cv_ran") == (sig, n_rep):
        st.session_state["cv_ran"] = (sig, n_rep)
        cv = run_cv(sig, n_rep)
        table, mean = cv_summary(cv, cutoff)
        st.markdown(f"**Mean ± std over {n_rep * cv['n_splits']} folds** (cutoff {cutoff:.2f})")
        st.dataframe(table, width="stretch", alt="Cross-validation mean and standard deviation per model")

        st.markdown("**McNemar's test**: do two models make significantly different mistakes?")
        st.dataframe(cv_significance(cv, cutoff), width="stretch", hide_index=True, alt="McNemar test results between model pairs")
        st.caption("Exact McNemar test on the out-of-fold predictions of each repeat (each message predicted once "
                   "per repeat); the median p-value across repeats is reported. p < 0.05 means the difference "
                   "is unlikely to be chance. Rows against the baseline test H1; rows between models test H2.")

        rec = recommend(cv, cutoff)
        b, r, m = rec["best"], rec["runner_up"], rec["mean"]
        lead = (f"Its lead over {r} is **statistically significant** (median p = {rec['p']:.4f})." if rec["p"] < 0.05
                else f"Its lead over {r} is **not statistically significant** (median p = {rec['p']:.4f}), "
                     "so either could be used.")
        cs = rec["cut_summary"]
        section("Recommendation")
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Model to deploy", b)
            k2.metric("Suggested cutoff", f"{rec['cutoff']:.2f}", help="Best F2 score: recall weighs twice "
                                                                        "as much as precision")
            k3.metric(f"CV recall at {cutoff:.2f}", f"{m.loc[b, 'Recall']:.3f}")
            k4.metric(f"CV F1 at {cutoff:.2f}", f"{m.loc[b, 'F1']:.3f}")
            st.markdown(
                f"{b} has the highest cross-validated recall at cutoff {cutoff:.2f}. {lead}  \n"
                f"At the suggested cutoff {rec['cutoff']:.2f}: recall {cs['Recall']:.3f}, precision "
                f"{cs['Precision']:.3f}, accuracy {cs['Accuracy']:.3f}. Random guessing reaches only "
                f"{m.loc[BASELINE, 'Accuracy']:.3f} accuracy.")
        st.caption("Calculated from the cross-validation above, so it updates when the data in sms_taglish.csv "
                   "or datasets/ changes. See the Feature analysis tab for which features matter most.")
    else:
        st.info("Select **Run cross-validation** to compute it. The result is kept until the data changes.",
                icon=":material/info:")

# ---------- Feature analysis ----------
with tab_feat:
    imp, names = art["importance"], art["feature_names"]
    n_feat = len(names)

    # --- Hand-made features across all models (RQ3) ---
    section("Hand-made features across all three models")
    rows = {}
    for mname, series in imp.items():
        order = series.sort_values(ascending=False)
        rank = pd.Series(np.arange(1, n_feat + 1), index=order.index)
        label = "importance" if mname == "Decision Tree" else "weight"
        rows[f"{mname} {label}"] = series[HAND_NAMES].round(3)
        rows[f"{mname} rank"] = rank[HAND_NAMES].where(series[HAND_NAMES].abs() >= 1e-3)  # unused -> blank
    hand_tbl = pd.DataFrame(rows)
    hand_tbl["Average rank"] = hand_tbl[[c for c in hand_tbl if c.endswith("rank")]].mean(axis=1).round(1)
    st.dataframe(hand_tbl.sort_values("Average rank"), width="stretch", alt="Hand-made feature weights and ranks per model")
    st.caption(f"Rank = position among all {n_feat:,} features (word/phrase TF-IDF + {len(HAND_NAMES)} hand-made), "
               "1 = strongest push toward SMISHING. LR/SVM weights are signed (negative = pushes toward legit); "
               "Decision Tree importance is unsigned (how much the tree relies on it). Blank rank = the model "
               "doesn't use that feature at all. sms_taglish.csv has no senders, so for its messages only "
               "`sender_unknown` is ever set.")

    # --- Permutation importance (model-agnostic, from CV) ---
    section("Permutation importance (from cross-validation)")
    n_rep = st.session_state.get("cv_repeats", 5)
    if st.session_state.get("cv_ran") == (sig, n_rep):
        perm = run_cv(sig, n_rep)["perm"]
        metric = st.segmented_control("Measure", ["Recall drop", "Accuracy drop"], default="Recall drop",
                                      required=True)
        ptbl = perm.pivot(index="Feature", columns="Model", values=metric)[MODEL_NAMES]
        ptbl = ptbl.loc[ptbl.mean(axis=1).sort_values(ascending=False).index]
        st.bar_chart(ptbl, horizontal=True, stack=False, sort=False, height=420,
                     alt=f"Bar chart of {metric.lower()} per hand-made feature for each model")
        st.dataframe(ptbl.round(4), width="stretch", alt="Permutation importance per feature and model")
        st.caption("How much each model's score drops when one hand-made feature is shuffled in the test folds "
                   "(5 shuffles x 5 folds). Bigger drop = the model relies on it more. Near zero = the model "
                   "doesn't need it, often because TF-IDF words already carry the same signal.")
    else:
        st.info("Run cross-validation in the **Model evaluation** tab first; this section uses the same run.",
                icon=":material/info:")

    # --- Feature sets (H3, Obj. 7) ---
    section("Feature sets: words, hand-made features, or both",
            f"Cross-validates every model again with only the TF-IDF words, and with only the {len(HAND_NAMES)} "
            f"hand-made features, on the same folds ({n_rep} repeats, as set in the Model evaluation tab). "
            "McNemar's test then checks each against words + hand-made.")
    if st.button("Run feature-set comparison", icon=":material/play_arrow:") \
            or st.session_state.get("fs_ran") == (sig, n_rep):
        st.session_state["fs_ran"] = (sig, n_rep)
        cvs = {fs: run_cv(sig, n_rep, fs) for fs in FEATURE_SETS}
        st.dataframe(feature_set_table(cvs, cutoff), width="stretch", alt="Feature-set comparison per model")
        st.caption(f"Cutoff {cutoff:.2f}. \"Significantly worse\" for words only means the hand-made features "
                   "add real information on top of the words; \"Not significantly different\" means the words "
                   "already carry the same signal.")
    else:
        st.info("Select **Run feature-set comparison** to compute it (it runs cross-validation twice more).",
                icon=":material/info:")

    # --- Each model's own view ---
    section("Top features per model")
    pick = st.selectbox("Model", MODEL_NAMES, key="feat_model")
    series = imp[pick]
    if pick == "Decision Tree":
        top = series.sort_values(ascending=False).head(15)
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.markdown("**Most used by the tree**")
            st.dataframe(top.round(4).rename("importance"), width="stretch")
        with c2:
            st.markdown("**Importance**")
            st.bar_chart(top.rename("Importance"), horizontal=True, sort=False, height=440,
                         alt="Bar chart of the Decision Tree's 15 most important features")
        st.markdown("**First 3 levels of the tree** (1 = smishing, 0 = legit; hand features are scaled to 0..1)")
        st.code(export_text(art["models"]["Decision Tree"], feature_names=names, max_depth=3,
                            show_weights=False), language="text")
    else:
        top = series.sort_values(ascending=False).head(15)
        c1, c2, c3 = st.columns([2, 2, 3], gap="medium")
        with c1:
            st.markdown("**Pushes toward smishing**")
            st.dataframe(top.round(3).rename("weight"), width="stretch")
        with c2:
            st.markdown("**Pushes toward legit**")
            st.dataframe(series.sort_values().head(15).round(3).rename("weight"), width="stretch")
        with c3:
            st.markdown("**Top smishing indicators**")
            st.bar_chart(top.rename("Weight"), horizontal=True, sort=False, height=440,
                         alt=f"Bar chart of the 15 features that push {pick} most toward smishing")

# ---------- Dataset ----------
with tab_data:
    df = base_art["df"]
    _, train_extra, summary, n_conf = load_data(sig)
    section("Core Filipino/Taglish dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Messages", f"{len(df):,}", border=True)
    d2.metric("Smishing", f"{int(df.label.sum()):,}", border=True)
    d3.metric("Legitimate", f"{int((df.label == 0).sum()):,}", border=True)
    st.write("`sms_taglish.csv` plus the `datasets/` files with role **core**, one row per template (the same "
             "message with a different OTP, amount or link counts once). The **Held-out test set** results use a "
             "fixed 80/20 split of it (the Analyze tab's model is trained on the 80%). **Cross-validation** "
             "rotates through all of it, so every message is tested once per repeat.")
    counts = df.assign(label=df.label.map({0: "Legit", 1: "Smishing"})).groupby(["source", "label"]).size()
    st.dataframe(counts.unstack(fill_value=0), width="stretch", alt="Smishing and legit counts per source")
    if n_conf:
        st.caption(f"{n_conf} rows were left out because the same template was labelled both smishing and legit.")
    st.markdown("**Sample rows**")
    st.dataframe(df.sample(min(10, len(df)), random_state=1), width="stretch", hide_index=True, alt="Ten sample rows of the core dataset")

    section("datasets/ folder")
    if summary.empty:
        st.info("No CSV or Excel files in `datasets/` yet. Drop Kaggle files there and refresh the page; "
                "see `datasets/README.md` for the format.", icon=":material/info:")
    else:
        st.write(f"**{(summary.Status == 'OK').sum()}** file(s) loaded. Role **core** files join the split above; "
                 f"role **train** files add **{len(train_extra):,}** unique messages to training only.")
        st.dataframe(summary, width="stretch", hide_index=True, alt="Per-file report for the datasets folder")
        if (~summary.Status.eq("OK") & ~summary.Status.str.startswith("Skipped: skipped in config")).any():
            st.warning("Some files were skipped. See the Status column, and `datasets/config.json` "
                       "(`datasets/README.md` explains it).")
