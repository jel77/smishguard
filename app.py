import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.tree import export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix)

import data_loader
from features import HAND_NAMES, clean, hand, sender_type

st.set_page_config(page_title="SmishGuard", page_icon="🛡️", layout="wide")

# ---------- Training ----------
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "SVM"]
BASELINE = "Random guess (baseline)"


def make_models():
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        # SVC(probability=True) is deprecated in scikit-learn 1.9; this gives the same calibrated probabilities
        "SVM": CalibratedClassifierCV(SVC(kernel="linear", class_weight="balanced", random_state=42),
                                      ensemble=False),
        # Guesses at random in the same smishing/legit ratio as the training data (H1 comparison)
        BASELINE: DummyClassifier(strategy="stratified", random_state=42),
    }


def add_extra(tr, te, extra):
    """Add `extra` rows to the training set, minus any that also appear in the test set
    (otherwise the scores would be inflated)."""
    if extra is None or not len(extra):
        return tr, 0
    test_texts = set(te.text.str.strip().str.lower())
    extra = extra[~extra.text.str.strip().str.lower().isin(test_texts)]
    return pd.concat([tr, extra[["text", "label", "sender"]]], ignore_index=True), len(extra)


def train_and_score(tr, te):
    """Fit the features and all models on `tr`.
    Returns (vec, sc, models, smishing probabilities on `te`, feature matrix of `te`)."""
    vec, sc = TfidfVectorizer(ngram_range=(1, 2)), MaxAbsScaler()

    def build(d, fit_=False):
        txt = d.text.apply(clean)
        H = np.array([hand(t, snd) for t, snd in zip(d.text, d.sender)])
        if fit_:
            return hstack([vec.fit_transform(txt), csr_matrix(sc.fit_transform(H))]).tocsr()
        return hstack([vec.transform(txt), csr_matrix(sc.transform(H))]).tocsr()

    Xtr, Xte = build(tr, True), build(te)
    models = make_models()
    probs = {n: m.fit(Xtr, tr.label).predict_proba(Xte)[:, 1] for n, m in models.items()}
    return vec, sc, models, probs, Xte


def metrics_table(y, probs, cutoff=0.5):
    """Metrics for each model when messages scoring >= `cutoff` are called smishing."""
    rows = []
    for name, p in probs.items():
        pred = (np.asarray(p) >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y, pred),
                     "Precision": precision_score(y, pred, zero_division=0),
                     "Recall": recall_score(y, pred, zero_division=0),
                     "F1": f1_score(y, pred, zero_division=0),
                     "Missed scams (FN)": int(fn),
                     "False alarms (FP)": int(fp)})
    return pd.DataFrame(rows).set_index("Model").round(3)


def fit(df, extra=None):
    """Train all models. `extra` = Kaggle datasets and/or flagged messages, added to the
    TRAINING set only, so the test set is always the held-out Taglish data."""
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    tr, n_extra = add_extra(tr, te, extra)
    vec, sc, models, probs, _ = train_and_score(tr, te)
    names = list(vec.get_feature_names_out()) + HAND_NAMES

    return {"vec": vec, "sc": sc, "models": {n: models[n] for n in MODEL_NAMES},
            "y_test": te.label.to_numpy(), "test_probs": probs,
            "importance": model_importance(models, names), "feature_names": names,
            "test_texts": set(te.text.str.strip().str.lower()),
            "n_train": len(tr), "n_test": len(te), "n_extra": n_extra, "df": df}


def model_importance(models, names):
    """Each model's own view of feature importance.
    LR and SVM: signed weights (+ pushes toward smishing, - toward legit; hand features are
    scaled to 0..1 like TF-IDF, so weights are comparable). DT: impurity importance (unsigned)."""
    svm = models["SVM"].calibrated_classifiers_[0].estimator
    svm_w = svm.coef_.toarray()[0] if hasattr(svm.coef_, "toarray") else np.asarray(svm.coef_)[0]
    return {"Logistic Regression": pd.Series(models["Logistic Regression"].coef_[0], index=names),
            "SVM": pd.Series(svm_w, index=names),
            "Decision Tree": pd.Series(models["Decision Tree"].feature_importances_, index=names)}


def permutation_drop(models, Xte, y, rng, n_shuffles=5):
    """For each hand-made feature: average drop in accuracy and recall when that feature's
    column is shuffled in the test fold (so the model can no longer use it)."""
    n_hand = len(HAND_NAMES)
    words, H = Xte[:, :-n_hand], Xte[:, -n_hand:].toarray()
    rows = []
    for name in MODEL_NAMES:
        m = models[name]
        base = m.predict(Xte)
        base_acc, base_rec = accuracy_score(y, base), recall_score(y, base, zero_division=0)
        for j, feat in enumerate(HAND_NAMES):
            acc, rec = [], []
            for _ in range(n_shuffles):
                Hp = H.copy()
                Hp[:, j] = rng.permutation(Hp[:, j])
                p = m.predict(hstack([words, csr_matrix(Hp)]).tocsr())
                acc.append(base_acc - accuracy_score(y, p))
                rec.append(base_rec - recall_score(y, p, zero_division=0))
            rows.append({"Model": name, "Feature": feat, "Accuracy drop": np.mean(acc), "Recall drop": np.mean(rec)})
    return pd.DataFrame(rows)


def mcnemar_p(y, pred_a, pred_b):
    """Exact McNemar test: do two models make significantly different mistakes on the same messages?"""
    a_right, b_right = pred_a == y, pred_b == y
    b, c = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def cross_validate(df, extra, n_repeats=5, n_splits=5):
    """Repeated stratified k-fold on the Taglish data. Every message is tested once per repeat;
    `extra` is added to each training fold only. Returns out-of-fold probabilities per repeat
    (so metrics can be recomputed at any cutoff without retraining)."""
    y = df.label.to_numpy()
    runs, perm = [], []
    rng = np.random.default_rng(0)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        oof = {n: np.zeros(len(df)) for n in make_models()}
        fold = np.zeros(len(df), dtype=int)
        for k, (a, b) in enumerate(skf.split(df, y)):
            tr, te = df.iloc[a], df.iloc[b]
            tr, _ = add_extra(tr, te, extra)
            _, _, models, probs, Xte = train_and_score(tr, te)
            for n, p in probs.items():
                oof[n][b] = p
            fold[b] = k
            if r == 0:  # permutation importance on the first repeat's folds
                perm.append(permutation_drop(models, Xte, te.label.to_numpy(), rng))
        runs.append({"oof": oof, "fold": fold})
    perm = pd.concat(perm).groupby(["Model", "Feature"], sort=False).mean().reset_index()
    return {"y": y, "runs": runs, "n_splits": n_splits, "perm": perm}


def cv_summary(cv, cutoff=0.5):
    """Mean ± std of each metric over all folds of all repeats."""
    y, per_fold = cv["y"], []
    for run in cv["runs"]:
        for k in range(cv["n_splits"]):
            m = run["fold"] == k
            per_fold.append(metrics_table(y[m], {n: p[m] for n, p in run["oof"].items()}, cutoff))
    stacked = pd.concat(per_fold)
    mean, std = stacked.groupby(level=0, sort=False).mean(), stacked.groupby(level=0, sort=False).std()
    cols = ["Accuracy", "Precision", "Recall", "F1"]
    table = pd.DataFrame({c: [f"{mean.loc[i, c]:.3f} ± {std.loc[i, c]:.3f}" for i in mean.index] for c in cols},
                         index=mean.index)
    return table, mean[cols]


def cv_significance(cv, cutoff=0.5):
    """McNemar test for every pair of models, run on each repeat's out-of-fold predictions."""
    y, names = cv["y"], list(cv["runs"][0]["oof"])
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ps = [mcnemar_p(y, run["oof"][a] >= cutoff, run["oof"][b] >= cutoff) for run in cv["runs"]]
            acc_a = np.mean([((run["oof"][a] >= cutoff) == y).mean() for run in cv["runs"]])
            acc_b = np.mean([((run["oof"][b] >= cutoff) == y).mean() for run in cv["runs"]])
            med = float(np.median(ps))
            rows.append({"Model A": a, "Model B": b,
                         "Accuracy A": round(acc_a, 3), "Accuracy B": round(acc_b, 3),
                         "Median p-value": round(med, 4),
                         "Significant repeats": f"{sum(p < 0.05 for p in ps)} of {len(ps)}",
                         "Verdict (p < 0.05)": ("Significant: " + (a if acc_a > acc_b else b) + " is better")
                                               if med < 0.05 else "Not significant"})
    return pd.DataFrame(rows)


def recommend(cv, cutoff=0.5):
    """Pick the model to deploy from the CV results: highest recall (missing a scam is the costly
    error), F1 as tie-breaker. Also checks the lead is significant and suggests a cutoff."""
    _, mean = cv_summary(cv, cutoff)
    ranked = mean.drop(BASELINE).sort_values(["Recall", "F1"], ascending=False)
    best, runner_up = ranked.index[0], ranked.index[1]
    sig = cv_significance(cv, cutoff)
    pair = sig[((sig["Model A"] == best) & (sig["Model B"] == runner_up)) |
               ((sig["Model A"] == runner_up) & (sig["Model B"] == best))].iloc[0]
    # Suggested cutoff: maximise F2 (weights recall twice as much as precision) on the out-of-fold predictions
    y, grid = cv["y"], np.round(np.arange(0.05, 0.96, 0.05), 2)
    f2 = [np.mean([fbeta_score(y, run["oof"][best] >= c, beta=2, zero_division=0) for run in cv["runs"]])
          for c in grid]
    top = max(f2)
    best_cut = min((c for c, v in zip(grid, f2) if v >= top - 1e-9), key=lambda c: abs(c - 0.5))
    return {"best": best, "runner_up": runner_up, "mean": mean, "p": pair["Median p-value"],
            "cutoff": float(best_cut), "cut_summary": cv_summary(cv, float(best_cut))[1].loc[best]}


@st.cache_resource(show_spinner="Loading dataset...")
def load_base(csv_path="sms_taglish.csv"):
    df = pd.read_csv(csv_path, dtype={"sender": str}).dropna(subset=["text", "label"])
    df["label"] = df.label.astype(int)
    df["sender"] = df["sender"].fillna("") if "sender" in df else ""
    return df


@st.cache_resource(show_spinner="Loading datasets/ folder...")
def load_extra(sig):
    """`sig` changes when files in datasets/ change, which reloads them."""
    return data_loader.load_extra()


@st.cache_resource(show_spinner="Training models...")
def base_artifacts(sig):
    return fit(load_base(), load_extra(sig)[0])


@st.cache_resource(show_spinner="Running cross-validation (this can take a while with big datasets)...")
def run_cv(sig, n_repeats):
    return cross_validate(load_base(), load_extra(sig)[0], n_repeats)


@st.cache_resource
def flag_store():
    """Shared in-memory list of flagged messages (all visitors share it).
    Lost when the app restarts, so download the CSV to keep it."""
    return {}


def predict_all(art, msg, sender=""):
    X = hstack([art["vec"].transform([clean(msg)]),
                csr_matrix(art["sc"].transform([hand(msg, sender)]))]).tocsr()
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
        in_test = fdf.text.str.strip().str.lower().isin(base_art["test_texts"])
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
    # --- Single held-out split ---
    st.subheader("Held-out test set")
    st.write(f"Train: **{art['n_train']}** messages (incl. **{art['n_extra']}** from datasets/ "
             f"and flags) | Test: **{art['n_test']}** Taglish messages | Cutoff: **{cutoff:.2f}**")
    results = metrics_table(art["y_test"], art["test_probs"], cutoff)
    st.dataframe(results, width="stretch")
    st.bar_chart(results[["Accuracy", "Precision", "Recall", "F1"]], stack=False)
    st.caption("Missed scams (FN) are the costly error here: a smishing text that got through. "
               f"With only {art['n_test']} test messages, one message moves accuracy by "
               f"{1 / art['n_test']:.1%}, so use the cross-validation below for the paper.")

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
    for model, series in imp.items():
        order = series.sort_values(ascending=False)
        rank = pd.Series(np.arange(1, n_feat + 1), index=order.index)
        label = "importance" if model == "Decision Tree" else "weight"
        rows[f"{model} {label}"] = series[HAND_NAMES].round(3)
        rows[f"{model} rank"] = rank[HAND_NAMES].where(series[HAND_NAMES].abs() >= 1e-3)  # unused -> blank
    hand_tbl = pd.DataFrame(rows)
    hand_tbl["Average rank"] = hand_tbl[[c for c in hand_tbl if c.endswith("rank")]].mean(axis=1).round(1)
    st.dataframe(hand_tbl.sort_values("Average rank"), width="stretch")
    st.caption(f"Rank = position among all {n_feat:,} features (word/phrase TF-IDF + {len(HAND_NAMES)} hand-made), "
               "1 = strongest push toward SMISHING. LR/SVM weights are signed (negative = pushes toward legit); "
               "Decision Tree importance is unsigned (how much the tree relies on it). Blank rank = the model "
               "doesn't use that feature at all. "
               "Sender features stay near zero until the `sender` column is filled in.")

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
    st.subheader("Core Taglish dataset (sms_taglish.csv)")
    st.write(f"{len(df)} messages. The **Held-out test set** results use a fixed 80/20 split of this file "
             "(the Live demo model is trained on the 80%). **Cross-validation** rotates through all of it, so "
             "every message is tested once per repeat. Files in `datasets/` are only ever used for training.")
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
