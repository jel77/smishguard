import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

import data_loader

st.set_page_config(page_title="SmishGuard", page_icon="🛡️", layout="wide")

# ---------- Features (same as the notebook) ----------
URGENT = ["kailangan agad", "agad", "ngayon", "claim now", "urgent", "expire", "last chance", "within 24"]
MONEY = ["php", "₱", "peso", "prize", "premyo", "panalo", "nanalo", "won", "reward", "cash", "gcash", "maya", "bdo", "bpi", "account"]
HAND_NAMES = ["has_url", "short_url", "urgency_count", "money_count", "length", "special_chars"]


def clean(t):
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+|bit\.ly\S*", " URL ", t)
    return re.sub(r"\d+", " NUM ", t)


def hand(t):
    l = t.lower()
    return [
        int(bool(re.search(r"https?://|www\.|\b[a-z0-9-]+\.(xyz|top|cc|info|online|net|com)\b", l))),
        int(bool(re.search(r"bit\.ly|tinyurl|cutt\.ly|shorturl", l))),
        sum(k in l for k in URGENT),
        sum(k in l for k in MONEY),
        len(t),
        sum(not c.isalnum() and not c.isspace() for c in t),
    ]


# ---------- Training ----------
def fit(df, extra=None):
    """Train all 3 models. `extra` = Kaggle datasets and/or flagged messages, added to the
    TRAINING set only, so the test set is always the held-out Taglish data."""
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    n_extra = 0
    if extra is not None and len(extra):
        # Drop extra rows that also appear in the test set, or the scores would be inflated
        test_texts = set(te.text.str.strip().str.lower())
        extra = extra[~extra.text.str.strip().str.lower().isin(test_texts)]
        n_extra = len(extra)
        tr = pd.concat([tr, extra[["text", "label"]]], ignore_index=True)

    vec = TfidfVectorizer(ngram_range=(1, 2))
    sc = MaxAbsScaler()

    def build(d, fit_=False):
        txt = d.text.apply(clean)
        H = np.array([hand(t) for t in d.text])
        if fit_:
            return hstack([vec.fit_transform(txt), csr_matrix(sc.fit_transform(H))]).tocsr()
        return hstack([vec.transform(txt), csr_matrix(sc.transform(H))]).tocsr()

    Xtr, Xte = build(tr, True), build(te)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    }

    rows = []
    for name, m in models.items():
        m.fit(Xtr, tr.label)
        p = m.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(te.label, p).ravel()
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(te.label, p),
                     "Precision": precision_score(te.label, p),
                     "Recall": recall_score(te.label, p),
                     "F1": f1_score(te.label, p),
                     "Missed scams (FN)": fn,
                     "False alarms (FP)": fp})
    results = pd.DataFrame(rows).round(3)

    names = list(vec.get_feature_names_out()) + HAND_NAMES
    coef = pd.Series(models["Logistic Regression"].coef_[0], index=names)

    return {"vec": vec, "sc": sc, "models": models, "results": results,
            "coef": coef, "n_train": len(tr), "n_test": len(te), "n_extra": n_extra, "df": df}


@st.cache_resource(show_spinner="Loading dataset...")
def load_base(csv_path="sms_taglish.csv"):
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df.label.astype(int)
    return df


@st.cache_resource(show_spinner="Loading datasets/ folder...")
def load_extra(sig):
    """`sig` changes when files in datasets/ change, which reloads them."""
    return data_loader.load_extra()


@st.cache_resource(show_spinner="Training models...")
def base_artifacts(sig):
    return fit(load_base(), load_extra(sig)[0])


@st.cache_resource
def flag_store():
    """Shared in-memory list of flagged messages (all visitors share it).
    Lost when the app restarts, so download the CSV to keep it."""
    return {}


def predict_all(art, msg):
    X = hstack([art["vec"].transform([clean(msg)]),
                csr_matrix(art["sc"].transform([hand(msg)]))]).tocsr()
    return {n: float(m.predict_proba(X)[0][1]) for n, m in art["models"].items()}


# ---------- UI ----------
st.title("🛡️ SmishGuard")
st.caption("Detecting banking and online payment smishing in Filipino/Taglish SMS")

sig = data_loader.signature()
try:
    base_art = base_artifacts(sig)
except FileNotFoundError:
    st.error("`sms_taglish.csv` not found. Put it in the same folder as app.py.")
    st.stop()

# Use the retrained model if this session has one
if st.session_state.get("retrained_sig", sig) != sig:
    st.session_state.pop("art_retrained", None)  # datasets/ changed: drop the stale retrain
art = st.session_state.get("art_retrained", base_art)
flags = flag_store()

tab_demo, tab_fb, tab_models, tab_feat, tab_data = st.tabs(
    ["🔎 Live demo", "🚩 Feedback", "📊 Model comparison", "🧠 Top features", "🗂️ Dataset"])

EXAMPLES = {
    "GCash scare + short link": "GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify",
    "Prize scam": "Congrats! Nanalo ka ng P30,000. I-claim now: promo-ph.xyz",
    "Real bank OTP": "Your BDO OTP is 482913. Do not share this code with anyone.",
    "Casual Taglish chat": "pre nasend ko na yung pambayad, check mo na lang",
}

with tab_demo:
    if "art_retrained" in st.session_state:
        st.info("Using the **retrained** model (includes flagged messages).")
    choice = st.selectbox("Try an example (or write your own below)", ["— custom —"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    msg = st.text_area("Paste an SMS message", value=default, height=120, key=f"msg_{choice}")

    if st.button("Analyze", type="primary") and msg.strip():
        st.session_state["last"] = {"msg": msg.strip(), "probs": predict_all(art, msg)}

    last = st.session_state.get("last")
    if last:
        probs = last["probs"]
        st.markdown(f"**Message:** {last['msg']}")
        cols = st.columns(3)
        for col, (name, p) in zip(cols, probs.items()):
            with col:
                st.subheader(name)
                if p >= 0.5:
                    st.error(f"⚠️ SMISHING ({p:.0%})")
                else:
                    st.success(f"✅ Legit ({p:.0%} smishing)")
                st.progress(min(max(p, 0.0), 1.0))
        votes = sum(p >= 0.5 for p in probs.values())
        st.info(f"{votes} of 3 models flag this message as smishing.")

        st.markdown("**Is the result wrong? Tell SmishGuard the correct label:**")
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("🚩 It's actually SMISHING"):
            flags[last["msg"]] = 1
            st.toast("Flagged as smishing. See the Feedback tab.")
        if b2.button("✅ It's actually LEGIT"):
            flags[last["msg"]] = 0
            st.toast("Flagged as legit. See the Feedback tab.")

with tab_fb:
    st.write("Messages you flag are collected here. Retrain to teach the models from them.")
    if not flags:
        st.info("Nothing flagged yet. Analyze a message in the Live demo tab, then use the flag buttons.")
    else:
        fdf = pd.DataFrame({"text": list(flags.keys()), "label": list(flags.values())})
        shown = fdf.assign(label=fdf.label.map({1: "Smishing", 0: "Legit"}))
        st.dataframe(shown, width="stretch")

        repeat = st.slider("Repeat each flagged message N times when training", 1, 20, 5,
                           help="One extra message barely moves a model trained on hundreds of rows, so repeating it gives it more weight.")
        c1, c2, c3 = st.columns(3)
        if c1.button("🔁 Retrain with flagged messages", type="primary"):
            extra = pd.concat([load_extra(sig)[0]] + [fdf] * repeat, ignore_index=True)
            with st.spinner("Retraining..."):
                st.session_state["art_retrained"] = fit(load_base(), extra)
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
    st.write(f"Train: **{art['n_train']}** messages (incl. **{art['n_extra']}** from datasets/ "
             f"and flags) | Test: **{art['n_test']}** Taglish messages")
    st.dataframe(art["results"].set_index("Model"), width="stretch")
    st.bar_chart(art["results"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])
    st.caption("Missed scams (FN) are the costly error here: a smishing text that got through. "
               "Test metrics always use the original held-out data.")

with tab_feat:
    coef = art["coef"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pushes toward SMISHING**")
        st.dataframe(coef.sort_values(ascending=False).head(15).round(3).rename("coef"))
    with c2:
        st.markdown("**Pushes toward LEGIT**")
        st.dataframe(coef.sort_values().head(15).round(3).rename("coef"))
    top = coef.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top[::-1].plot(kind="barh", color="#c0392b", ax=ax)
    ax.set_title("Top smishing indicators (Logistic Regression coefficients)")
    fig.tight_layout()
    st.pyplot(fig)

with tab_data:
    df = base_art["df"]
    st.subheader("Core Taglish dataset (sms_taglish.csv)")
    st.write(f"{len(df)} messages. 80% is used for training, 20% is the test set for every score in the app.")
    st.bar_chart(df.label.map({0: "Legit", 1: "Smishing"}).value_counts())
    st.dataframe(df.sample(min(10, len(df)), random_state=1), width="stretch")

    st.subheader("Extra training data (datasets/ folder)")
    extra, summary = load_extra(sig)
    if summary.empty:
        st.info("No CSVs in `datasets/` yet. Drop Kaggle CSVs there and refresh the page; "
                "see `datasets/README.md` for the format.")
    else:
        st.write(f"**{len(extra)}** unique messages from **{(summary.Status == 'OK').sum()}** file(s), "
                 "used for training only.")
        st.dataframe(summary, width="stretch", hide_index=True)
        if (summary.Status != "OK").any():
            st.warning("Some files were skipped. Fix them with `datasets/config.json` (see `datasets/README.md`).")
