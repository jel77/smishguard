"""Training, evaluation and statistics for SmishGuard, kept apart from the Streamlit UI (app.py)
so it can be tested and run without the app."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix)

import data_loader
from features import HAND_NAMES, clean, hand

BASE_CSV = Path(__file__).parent / "sms_taglish.csv"

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "SVM"]
BASELINE = "Random guess (baseline)"
# Which features the models see (H3: do the hand-made features add anything over the words?)
FEATURE_SETS = {"both": "Words + hand-made", "words": "Words only (TF-IDF)", "hand": "Hand-made only"}


def make_models():
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        # Linear SVM. LinearSVC (liblinear) instead of SVC(kernel="linear") (libsvm): same kind of model,
        # ~50x faster on thousands of messages. Calibration gives it smishing probabilities
        "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), ensemble=False),
        # Guesses at random in the same smishing/legit ratio as the training data (H1 comparison)
        BASELINE: DummyClassifier(strategy="stratified", random_state=42),
    }


def add_extra(tr, te, extra):
    """Add `extra` rows to the training set, minus any whose template also appears in the test set
    (otherwise the scores would be inflated)."""
    if extra is None or not len(extra):
        return tr, 0
    test_keys = set(te.text.map(data_loader.dedup_key))
    extra = extra[~extra.text.map(data_loader.dedup_key).isin(test_keys)]
    return pd.concat([tr, extra[["text", "label", "sender"]]], ignore_index=True), len(extra)


def strata(df):
    """Split so each source keeps its smishing/legit ratio in both train and test."""
    return df.source.astype(str) + "|" + df.label.astype(str)


def hide_senders(tr):
    """Sender dropout: every training message that has a sender is added a second time without it.
    Otherwise "no sender" is learned as a sign of legit (most sender-less training rows are legit),
    and smishing in sms_taglish.csv, which has no senders, gets missed."""
    with_sender = tr[tr.sender.astype(str).str.strip() != ""]
    return pd.concat([tr, with_sender.assign(sender="")], ignore_index=True)


def train_and_score(tr, te, features="both"):
    """Fit the features and all models on `tr`. `features` is a key of FEATURE_SETS.
    Returns (vec, sc, models, smishing probabilities on `te`, feature matrix of `te`)."""
    tr = hide_senders(tr)
    vec, sc = TfidfVectorizer(ngram_range=(1, 2)), MaxAbsScaler()

    def build(d, fit_=False):
        parts = []
        if features in ("both", "words"):
            txt = d.text.apply(clean)
            parts.append(vec.fit_transform(txt) if fit_ else vec.transform(txt))
        if features in ("both", "hand"):
            H = np.array([hand(t, snd) for t, snd in zip(d.text, d.sender)])
            parts.append(csr_matrix(sc.fit_transform(H) if fit_ else sc.transform(H)))
        return hstack(parts).tocsr()

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
    """Train all models. `df` = the core Filipino/Taglish set, split 80/20. `extra` = training-only
    datasets and/or flagged messages, added to the TRAINING set only, so the test set is always
    held-out core data."""
    tr, te = train_test_split(df, test_size=0.2, stratify=strata(df), random_state=42)
    tr, n_extra = add_extra(tr, te, extra)
    vec, sc, models, probs, _ = train_and_score(tr, te)
    # A TF-IDF word can share a name with a hand-made feature (the word "length"); keep names unique
    names = [f"{w} (word)" if w in HAND_NAMES else w for w in vec.get_feature_names_out()] + HAND_NAMES

    return {"vec": vec, "sc": sc, "models": {n: models[n] for n in MODEL_NAMES},
            "y_test": te.label.to_numpy(), "test_probs": probs, "test_source": te.source.to_numpy(),
            "importance": model_importance(models, names), "feature_names": names,
            "test_keys": set(te.text.map(data_loader.dedup_key)),
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


def cross_validate(df, extra, n_repeats=5, n_splits=5, features="both"):
    """Repeated stratified k-fold on the core Filipino/Taglish set. Every message is tested once per
    repeat; `extra` is added to each training fold only. Returns out-of-fold probabilities per repeat
    (so metrics can be recomputed at any cutoff without retraining). The folds depend only on the
    data and the repeat number, so runs with different `features` are paired message by message.
    Permutation importance needs the hand-made features, so it is only run for features="both"."""
    y = df.label.to_numpy()
    runs, perm = [], []
    rng = np.random.default_rng(0)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        oof = {n: np.zeros(len(df)) for n in make_models()}
        fold = np.zeros(len(df), dtype=int)
        for k, (a, b) in enumerate(skf.split(df, strata(df))):
            tr, te = df.iloc[a], df.iloc[b]
            tr, _ = add_extra(tr, te, extra)
            _, _, models, probs, Xte = train_and_score(tr, te, features)
            for n, p in probs.items():
                oof[n][b] = p
            fold[b] = k
            if r == 0 and features == "both":  # permutation importance on the first repeat's folds
                perm.append(permutation_drop(models, Xte, te.label.to_numpy(), rng))
        runs.append({"oof": oof, "fold": fold})
    perm = pd.concat(perm).groupby(["Model", "Feature"], sort=False).mean().reset_index() if perm else None
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


def feature_set_table(cvs, cutoff=0.5):
    """`cvs` = {feature set: cross_validate result}, all run on the same folds. For each model and
    feature set: mean CV recall and F1, and a McNemar test against "both" (median p over repeats)."""
    y, rows = cvs["both"]["y"], []
    for name in MODEL_NAMES:
        for fs, cv in cvs.items():
            _, mean = cv_summary(cv, cutoff)
            row = {"Model": name, "Features": FEATURE_SETS[fs],
                   "Recall": round(mean.loc[name, "Recall"], 3), "F1": round(mean.loc[name, "F1"], 3)}
            if fs != "both":
                ps = [mcnemar_p(y, ra["oof"][name] >= cutoff, rb["oof"][name] >= cutoff)
                      for ra, rb in zip(cvs["both"]["runs"], cv["runs"])]
                med = float(np.median(ps))
                worse = mean.loc[name, "F1"] < cv_summary(cvs["both"], cutoff)[1].loc[name, "F1"]
                row["p vs words + hand-made"] = round(med, 4)
                row["Verdict (p < 0.05)"] = (("Significantly worse" if worse else "Significantly better")
                                             if med < 0.05 else "Not significantly different")
            rows.append(row)
    return pd.DataFrame(rows).set_index(["Model", "Features"])


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


def load_base(csv_path=BASE_CSV):
    df = pd.read_csv(csv_path, dtype={"sender": str}).dropna(subset=["text", "label"])
    df["label"] = df.label.astype(int)
    df["sender"] = df["sender"].fillna("") if "sender" in df else ""
    df["source"] = Path(csv_path).name
    return df[["text", "label", "sender", "source"]]

def load_data(data_dir=data_loader.DATA_DIR, csv_path=BASE_CSV):
    """Returns (core, train_extra, summary, n_conflicts). core = sms_taglish.csv + files with role
    "core", one row per template; it is what gets split and cross-validated. train_extra = files
    with role "train", used for training only. summary = the per-file report from data_loader."""
    extra, summary = data_loader.load_extra(data_dir)
    core, n_conf = data_loader.dedupe(pd.concat([load_base(csv_path), extra[extra.role == "core"]],
                                                ignore_index=True))
    train_extra, n_conf_tr = data_loader.dedupe(extra[extra.role == "train"])
    return core.drop(columns="role", errors="ignore"), train_extra, summary, n_conf + n_conf_tr


def predict_all(art, msg, sender=""):
    X = hstack([art["vec"].transform([clean(msg)]),
                csr_matrix(art["sc"].transform([hand(msg, sender)]))]).tocsr()
    return {n: float(m.predict_proba(X)[0][1]) for n, m in art["models"].items()}
