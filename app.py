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


# ---------- Train once, cache across reruns and users ----------
@st.cache_resource(show_spinner="Training models...")
def train(csv_path="sms_taglish.csv"):
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df.label.astype(int)

    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    sc = MaxAbsScaler()

    def build(d, fit=False):
        txt = d.text.apply(clean)
        H = np.array([hand(t) for t in d.text])
        if fit:
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
            "coef": coef, "n_train": len(tr), "n_test": len(te), "df": df}


def predict_all(art, msg):
    X = hstack([art["vec"].transform([clean(msg)]),
                csr_matrix(art["sc"].transform([hand(msg)]))]).tocsr()
    return {n: float(m.predict_proba(X)[0][1]) for n, m in art["models"].items()}


# ---------- UI ----------
st.title("🛡️ SmishGuard")
st.caption("Detecting banking and online payment smishing in Filipino/Taglish SMS")

try:
    art = train()
except FileNotFoundError:
    st.error("`sms_taglish.csv` not found. Put it in the same folder as app.py.")
    st.stop()

tab_demo, tab_models, tab_feat, tab_data = st.tabs(
    ["🔎 Live demo", "📊 Model comparison", "🧠 Top features", "🗂️ Dataset"])

EXAMPLES = {
    "GCash scare + short link": "GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify",
    "Prize scam": "Congrats! Nanalo ka ng P30,000. I-claim now: promo-ph.xyz",
    "Real bank OTP": "Your BDO OTP is 482913. Do not share this code with anyone.",
    "Casual Taglish chat": "pre nasend ko na yung pambayad, check mo na lang",
}

with tab_demo:
    choice = st.selectbox("Try an example (or write your own below)", ["— custom —"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    msg = st.text_area("Paste an SMS message", value=default, height=120, key=f"msg_{choice}")

    if st.button("Analyze", type="primary") and msg.strip():
        probs = predict_all(art, msg)
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

with tab_models:
    st.write(f"Train: **{art['n_train']}** messages | Test: **{art['n_test']}** messages")
    st.dataframe(art["results"].set_index("Model"), width="stretch")
    st.bar_chart(art["results"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])
    st.caption("Missed scams (FN) are the costly error here: a smishing text that got through.")

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
    df = art["df"]
    st.write(df.shape)
    st.bar_chart(df.label.map({0: "Legit", 1: "Smishing"}).value_counts())
    st.dataframe(df.sample(min(10, len(df)), random_state=1), width="stretch")
