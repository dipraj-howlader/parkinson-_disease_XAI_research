#!/usr/bin/env python3
# Figure 1 only (participant flow) - extracted from pd_revised_analysis_v2.py with preprocessing fix applied
import warnings
warnings.filterwarnings("ignore")

import argparse, json, re, sys
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, fisher_exact, spearmanr, binomtest

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
try:
    from xgboost import XGBClassifier
    XGB_REAL = True
except ImportError:   # fallback ONLY for sandbox without xgboost
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_REAL = False
    def XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.2, eval_metric=None, random_state=0, verbosity=0):
        return GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=learning_rate, random_state=random_state)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="pd_speech_features.csv")
ap.add_argument("--out", default="results_v2")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

CSV_PATH = args.csv
OUT = Path(args.out); OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
N_SPLITS_CV = 5
TEST_SIZE = 0.20
N_BOOT = 50 if args.quick else 1000
N_REPEATS = 2 if args.quick else 20
MISSING_THRESHOLD = 0.10
THR = 0.50
TOP_K = 20
MODEL_ORDER = ["KNN", "XGBoost", "RandomForest", "SVM", "NaiveBayes"]
COLORS = {"KNN": "#2196F3", "XGBoost": "#FF9800", "RandomForest": "#4CAF50",
          "SVM": "#9C27B0", "NaiveBayes": "#F44336"}
VALUES = {}          # everything the manuscript builder needs
np.random.seed(RANDOM_STATE)


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def fmt(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if np.isnan(x) else float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, (set, tuple)): return list(x)
    return str(x)


# ----------------------------------------------------------------------------
# 1  DATA
# ----------------------------------------------------------------------------
banner("1  DATA LOADING")
raw = pd.read_csv(CSV_PATH, header=None, dtype=str)
hdr = next((i for i in range(min(10, len(raw)))
            if "class" in raw.iloc[i].astype(str).str.lower().str.strip().tolist()), 0)
df = raw.iloc[hdr + 1:].reset_index(drop=True).copy()
df.columns = raw.iloc[hdr].astype(str).str.strip().tolist()
class_col = next(c for c in df.columns if c.lower() == "class")
id_col = next((c for c in df.columns if c.lower() in ("id", "subject", "speaker")), df.columns[0])
gender_col = next((c for c in df.columns if c.lower() in ("gender", "sex")), None)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
struct = {id_col, class_col} | ({gender_col} if gender_col else set())
feature_cols = [c for c in df.columns if c not in struct]
n_cols_total = df.shape[1]

# exact missing-value accounting  (R1.4)
miss_cells = int(df[feature_cols].isna().sum().sum())
miss_feats = int((df[feature_cols].isna().sum() > 0).sum())
miss_rows = int(df[feature_cols].isna().any(axis=1).sum())
miss_frac = df[feature_cols].isna().mean()
dropped = miss_frac[miss_frac > MISSING_THRESHOLD].index.tolist()
feature_cols = [c for c in feature_cols if c not in dropped]
n_const = int((df[feature_cols].nunique(dropna=True) <= 1).sum())
pd.Series(feature_cols, name="feature").to_csv(OUT / "feature_columns_used.csv", index=False)

X = df[feature_cols].copy()
y = df[class_col].astype(int)
groups = df[id_col]
sex = df[gender_col].astype(int) if gender_col else None
subj = df.groupby(id_col).agg(cls=(class_col, "first"),
                              sex=(gender_col, "first")).reset_index()
subj.columns = ["subject_id", "cls", "sex"]
recs_per = df.groupby(id_col).size()

VALUES["dataset"] = dict(
    n_recordings=len(df), n_subjects=int(df[id_col].nunique()),
    n_columns_total=n_cols_total, n_structural=len(struct),
    structural=sorted(struct), n_features=len(feature_cols),
    n_features_dropped=len(dropped), missing_cells=miss_cells,
    missing_features=miss_feats, missing_rows=miss_rows,
    constant_features=n_const,
    recs_per_subject_min=int(recs_per.min()), recs_per_subject_max=int(recs_per.max()),
    n_pd=int((subj.cls == 1).sum()), n_ctrl=int((subj.cls == 0).sum()),
    pd_male=int(((subj.cls == 1) & (subj.sex == 1)).sum()),
    pd_female=int(((subj.cls == 1) & (subj.sex == 0)).sum()),
    ctrl_male=int(((subj.cls == 0) & (subj.sex == 1)).sum()),
    ctrl_female=int(((subj.cls == 0) & (subj.sex == 0)).sum()),
    prevalence=float((subj.cls == 1).mean()))
print(json.dumps(VALUES["dataset"], indent=1, default=fmt))


# ----------------------------------------------------------------------------
# 2  SPLITS
# ----------------------------------------------------------------------------
def make_split(seed):
    """Stratified, participant-level 80/20 split (identical to v1 for seed 42)."""
    rng = np.random.RandomState(seed)
    test = []
    for c in [0, 1]:
        s = subj[subj.cls == c]["subject_id"].tolist()
        rng.shuffle(s)
        test.extend(s[:max(1, round(len(s) * TEST_SIZE))])
    return set(test)


banner("2  TRAIN / HOLDOUT SPLIT (participant-disjoint)")
test_set = make_split(RANDOM_STATE)
train_set = set(subj.subject_id) - test_set
assert not (test_set & train_set)
tr_m, te_m = groups.isin(train_set), groups.isin(test_set)


def part(m):
    d = df[m].reset_index(drop=True)
    return (d[feature_cols], d[class_col].astype(int), d[id_col],
            d[gender_col].astype(int) if gender_col else None)


Xtr, ytr, gtr, str_ = part(tr_m)
Xte, yte, gte, ste = part(te_m)
sp = subj.assign(partition=np.where(subj.subject_id.isin(test_set), "holdout", "train"))
sp.to_csv(OUT / "split_assignment_participants.csv", index=False)


def cnt(sel):
    return dict(n=len(sel), pd=int((sel.cls == 1).sum()), ctrl=int((sel.cls == 0).sum()),
                male=int((sel.sex == 1).sum()), female=int((sel.sex == 0).sum()),
                pd_male=int(((sel.cls == 1) & (sel.sex == 1)).sum()),
                pd_female=int(((sel.cls == 1) & (sel.sex == 0)).sum()),
                ctrl_male=int(((sel.cls == 0) & (sel.sex == 1)).sum()),
                ctrl_female=int(((sel.cls == 0) & (sel.sex == 0)).sum()))


VALUES["split"] = dict(seed=RANDOM_STATE, overlap=0, train_recordings=len(Xtr),
                       test_recordings=len(Xte),
                       train=cnt(sp[sp.partition == "train"]),
                       holdout=cnt(sp[sp.partition == "holdout"]))
print(json.dumps(VALUES["split"], indent=1))


# ----------------------------------------------------------------------------
# 3  MODELS
# ----------------------------------------------------------------------------
class SexResidualizer(BaseEstimator, TransformerMixin):
    """The LAST column is sex (0/1).  Fit on training data only: per-feature
    sex-specific means are estimated and subtracted (= residuals of an OLS
    feature ~ sex); the sex column is then dropped.  Used for sensitivity."""
    def fit(self, X, y=None):
        X = np.asarray(X, float); s = X[:, -1]; F = X[:, :-1]
        self.m0_ = F[s == 0].mean(0) if (s == 0).any() else F.mean(0)
        self.m1_ = F[s == 1].mean(0) if (s == 1).any() else F.mean(0)
        return self

    def transform(self, X):
        X = np.asarray(X, float); s = X[:, -1][:, None]; F = X[:, :-1]
        return F - np.where(s == 1, self.m1_, self.m0_)


def make_models(sex_adjust=False):
    imp = lambda: ("imputer", SimpleImputer(strategy="median"))
    res = [("sexres", SexResidualizer())] if sex_adjust else []
    sc = lambda: ("scaler", StandardScaler())
    return {
        "KNN": Pipeline([imp(), *res, sc(), ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean"))]),
        "XGBoost": Pipeline([imp(), *res, ("clf", XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.2, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0))]),
        "RandomForest": Pipeline([imp(), *res, ("clf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE))]),
        "SVM": Pipeline([imp(), *res, sc(), ("clf", SVC(kernel="rbf", probability=True,
                                                         random_state=RANDOM_STATE))]),
        "NaiveBayes": Pipeline([imp(), *res, sc(), ("clf", GaussianNB())]),
    }


def with_sex(Xd, s):
    Z = Xd.copy(); Z["__sex__"] = np.asarray(s); return Z


# ----------------------------------------------------------------------------
# 4  METRIC HELPERS
# ----------------------------------------------------------------------------
def subject_aggregate(g, yt, yp, thr=THR):
    d = pd.DataFrame({"group": g, "y_true": yt, "y_prob": yp})
    s = d.groupby("group").agg(y_true=("y_true", "max"), y_prob=("y_prob", "mean")).reset_index()
    s["y_pred"] = (s.y_prob >= thr).astype(int)
    return s


def cp_ci(k, n, a=0.05):
    if n == 0: return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def metrics(yt, yp_b, yprob):
    tn, fp, fn, tp = confusion_matrix(yt, yp_b, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(yt, yp_b),
                Precision=precision_score(yt, yp_b, zero_division=0),
                Recall=recall_score(yt, yp_b, zero_division=0),
                Specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                F1=f1_score(yt, yp_b, zero_division=0),
                AUROC=roc_auc_score(yt, yprob), AUPRC=average_precision_score(yt, yprob),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def _spec(yt, yp):
    neg = yt == 0
    return np.nan if neg.sum() == 0 else float(((yp < THR) & neg).sum() / neg.sum())


MET_FN = {
    "Accuracy": lambda yt, yp: accuracy_score(yt, (yp >= THR).astype(int)),
    "Precision": lambda yt, yp: precision_score(yt, (yp >= THR).astype(int), zero_division=0),
    "Recall": lambda yt, yp: recall_score(yt, (yp >= THR).astype(int), zero_division=0),
    "Specificity": _spec,
    "F1": lambda yt, yp: f1_score(yt, (yp >= THR).astype(int), zero_division=0),
    "AUROC": roc_auc_score, "AUPRC": average_precision_score}


def boot_ci(fn, yt, yp, n_boot=None, seed=RANDOM_STATE):
    n_boot = n_boot or N_BOOT
    r = np.random.RandomState(seed); n = len(yt); st = []
    for _ in range(n_boot):
        i = r.randint(0, n, n)
        try: st.append(fn(yt[i], yp[i]))
        except Exception: st.append(np.nan)
    st = np.array(st, float)
    return float(np.nanpercentile(st, 2.5)), float(np.nanpercentile(st, 97.5))


def summarise(res, tag):
    """res: name -> subject-level DataFrame.  Returns list of row dicts."""
    rows = []
    for n, d in res.items():
        yt, yp, yb = d.y_true.values, d.y_prob.values, d.y_pred.values
        m = metrics(yt, yb, yp); row = dict(Model=n, n_subjects=len(d), set=tag)
        for k in ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]:
            lo, hi = boot_ci(MET_FN[k], yt, yp)
            row[k] = m[k]; row[k + "_lo"] = lo; row[k + "_hi"] = hi
        row["Specificity"] = m["Specificity"]
        row["Specificity_lo"], row["Specificity_hi"] = cp_ci(m["TN"], m["TN"] + m["FP"])
        row["Sensitivity_CP_lo"], row["Sensitivity_CP_hi"] = cp_ci(m["TP"], m["TP"] + m["FN"])
        for k in ["TP", "FP", "TN", "FN"]: row[k] = m[k]
        rows.append(row)
    return rows


def fit_predict(models, Xa, ya, Xb, gb):
    out = {}; rec = {}
    for n, p in models.items():
        p.fit(Xa, ya); pr = p.predict_proba(Xb)[:, 1]; rec[n] = pr
    return rec


# ----------------------------------------------------------------------------
# 5  CROSS-VALIDATION (training set only) + model selection by CV AUROC
# ----------------------------------------------------------------------------
banner("5  5-FOLD GroupKFold CV (training participants only)")
MODELS = make_models()
gkf = GroupKFold(n_splits=N_SPLITS_CV)
folds = list(gkf.split(Xtr, ytr, groups=gtr))
cv_rec = {n: [] for n in MODEL_ORDER}
oof = {n: [] for n in MODEL_ORDER}
for k, (a, b) in enumerate(folds):
    assert not (set(gtr.iloc[a]) & set(gtr.iloc[b]))
    for n in MODEL_ORDER:
        MODELS[n].fit(Xtr.iloc[a], ytr.iloc[a])
        pr = MODELS[n].predict_proba(Xtr.iloc[b])[:, 1]
        d = subject_aggregate(gtr.iloc[b].values, ytr.iloc[b].values, pr)
        m = metrics(d.y_true.values, d.y_pred.values, d.y_prob.values); m["fold"] = k + 1
        m["n_val_subjects"] = len(d); cv_rec[n].append(m); d["fold"] = k + 1; oof[n].append(d)
    print(f"  fold {k+1}: " + "  ".join(f"{n}={cv_rec[n][-1]['AUROC']:.3f}" for n in MODEL_ORDER))
cv_rows = []
for n in MODEL_ORDER:
    f = pd.DataFrame(cv_rec[n]); f.insert(0, "Model", n)
    f.to_csv(OUT / f"cv_folds_{n}.csv", index=False)
    r = {"Model": n}
    for k in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUROC", "AUPRC"]:
        r[k + "_mean"] = f[k].mean(); r[k + "_sd"] = f[k].std()
    cv_rows.append(r)
cv_df = pd.DataFrame(cv_rows); cv_df.to_csv(OUT / "cv_summary.csv", index=False)
PRIMARY = cv_df.sort_values("AUROC_mean", ascending=False).iloc[0]["Model"]
print(f"\n  Pre-specified selection rule: highest mean CV AUROC (training data only)\n  -> PRIMARY MODEL = {PRIMARY}")

# participant flow figure
fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off")
ds_ = VALUES["dataset"]; sp_ = VALUES["split"]
boxes = [(.5, .88, f"PD Speech Features dataset\n{ds_['n_subjects']} participants ({ds_['n_pd']} PD, {ds_['n_ctrl']} controls)\n{ds_['n_recordings']} recordings (3 per participant)"),
         (.25, .52, f"Training partition\n{sp_['train']['n']} participants ({sp_['train']['pd']} PD / {sp_['train']['ctrl']} ctrl)\n{sp_['train_recordings']} recordings"),
         (.75, .52, f"Holdout partition (used once)\n{sp_['holdout']['n']} participants ({sp_['holdout']['pd']} PD / {sp_['holdout']['ctrl']} ctrl)\n{sp_['test_recordings']} recordings\nFinal fit on training partition only\n(imputer/scaler fitted on training data)"),
         (.25, .15, f"5-fold GroupKFold CV (participant-grouped)\nImputer + scaler fitted on each training fold only\n-> model selection by mean CV AUROC = {PRIMARY}")]
for x_, y_, tx in boxes:
    ax.text(x_, y_, tx, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=.5", fc="#EAF1FB", ec="#4C78A8"))
for (x1, y1, x2, y2) in [(.5, .78, .25, .62), (.5, .78, .75, .62), (.25, .42, .25, .25)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))
ax.text(.5, .52, "no participant overlap", ha="center", fontsize=9, style="italic")
plt.savefig(OUT / "fig1_participant_flow.png", dpi=300, bbox_inches="tight"); plt.close()
