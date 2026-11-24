#!/usr/bin/env python3
# =============================================================================
#  PD SPEECH FEATURE ANALYSIS  -  VERSION 2  (second-round revision)
#  Participant-level machine-learning benchmark, PLOS ONE
# =============================================================================
#  What is new relative to pd_revised_analysis.py (v1)
#  ---------------------------------------------------
#  [S1] Specificity (+ exact Clopper-Pearson CI) reported next to sensitivity.
#  [S2] Majority-class baseline and sex-only baseline (what do acoustics add?).
#  [S3] Model selection uses TRAINING-SET CV AUROC only (pre-specified); the
#       holdout is evaluated once.  Pooled out-of-fold subject-level metrics
#       (n = 201 subjects) are added.
#  [S4] Paired bootstrap on the SAME holdout subjects: differences between
#       models (vs. the CV-selected model) with 95 % CI, plus exact McNemar.
#  [S5] SHAP re-done with exact TreeExplainer on the CV-selected tree model
#       (v1 used PermutationExplainer with max_evals = 2M+1 = ONE permutation,
#       which gives noisy rankings).  All holdout recordings are explained,
#       signed direction + subject-level bootstrap CI, feature-FAMILY
#       importance, stability of the top features across CV folds, and local
#       (waterfall) examples for TP / TN / FP / FN participants.
#  [S6] Sex analyses: (a) stratified metrics with exact CIs and Fisher tests,
#       (b) can acoustic features predict sex (proxy check)?, (c) sensitivity
#       analysis in which the sex-specific mean of every feature is removed
#       INSIDE each training fold (SexResidualizer).
#  [S7] Robustness to the single small holdout: 20 repeated stratified
#       participant-level splits (descriptive; training sets overlap).
#  [S8] Reproducibility: exact feature list, missing-value counts, train/test
#       participant IDs, recording-level holdout predictions, software
#       versions and a manuscript_values.json used to build the manuscript.
#
#  USAGE
#      python pd_revised_analysis_v2.py --csv pd_speech_features.csv --out results_v2
#      (add --quick for a fast smoke test with fewer bootstrap/repeat runs)
# =============================================================================
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

import sklearn, xgboost, shap
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
from xgboost import XGBClassifier

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
VALUES["cv"] = cv_df.to_dict(orient="records"); VALUES["primary_model"] = PRIMARY
# S2 table: per-fold file for supplement
pd.concat([pd.DataFrame(cv_rec[n]).assign(Model=n) for n in MODEL_ORDER]).to_csv(
    OUT / "S2_cv_per_fold_all_models.csv", index=False)

# pooled out-of-fold, subject level (n = 201)
pooled = {n: pd.concat(oof[n]).reset_index(drop=True) for n in MODEL_ORDER}
pooled_rows = summarise(pooled, "pooled_oof_cv")
pd.DataFrame(pooled_rows).to_csv(OUT / "cv_pooled_oof_summary.csv", index=False)
VALUES["cv_pooled"] = pooled_rows

# ----------------------------------------------------------------------------
# 6  FINAL FIT + HOLDOUT (evaluated once)
# ----------------------------------------------------------------------------
banner("6  FINAL FIT ON TRAINING SET + HOLDOUT EVALUATION")
MODELS = make_models()
rec_prob = fit_predict(MODELS, Xtr, ytr, Xte, gte)
hold = {n: subject_aggregate(gte.values, yte.values, rec_prob[n]) for n in MODEL_ORDER}
ref_ids = hold[MODEL_ORDER[0]].group.values
for n in MODEL_ORDER:
    assert (hold[n].group.values == ref_ids).all()          # same participants everywhere
    hold[n].to_csv(OUT / f"holdout_subject_predictions_{n}.csv", index=False)
pd.DataFrame({"id": gte.values, "class": yte.values, "sex": ste.values,
              **{n: rec_prob[n] for n in MODEL_ORDER}}).to_csv(
    OUT / "holdout_recording_predictions.csv", index=False)

rows = summarise(hold, "holdout")
# baselines
yt_h = hold[PRIMARY].y_true.values
maj = dict(Model="Majority class (all PD)", n_subjects=len(yt_h), set="baseline",
           Accuracy=float(yt_h.mean()), Precision=float(yt_h.mean()), Recall=1.0,
           Specificity=0.0, F1=2 * yt_h.mean() / (1 + yt_h.mean()), AUROC=0.5,
           AUPRC=float(yt_h.mean()), TP=int(yt_h.sum()), FP=int((1 - yt_h).sum()), TN=0, FN=0)
subj_sex_tr = subj[subj.subject_id.isin(train_set)]
lr = LogisticRegression().fit(subj_sex_tr[["sex"]], subj_sex_tr.cls)
sx_te = subj.set_index("subject_id").loc[ref_ids, "sex"].values
sex_only = pd.DataFrame({"group": ref_ids, "y_true": yt_h, "y_prob": lr.predict_proba(sx_te[:, None])[:, 1]})
sex_only["y_pred"] = (sex_only.y_prob >= THR).astype(int)
so = summarise({"Sex-only logistic baseline": sex_only}, "baseline")[0]
rows += [maj, so]
tab1 = pd.DataFrame(rows); tab1.to_csv(OUT / "table_holdout_subject_level.csv", index=False)
VALUES["holdout"] = rows
for r in rows:
    print(f"  {r['Model']:28s} AUROC={r['AUROC']:.3f} Acc={r['Accuracy']:.3f} "
          f"Sens={r['Recall']:.3f} Spec={r['Specificity']:.3f}  TP={r['TP']} FP={r['FP']} TN={r['TN']} FN={r['FN']}")

# ----------------------------------------------------------------------------
# 7  PAIRED BOOTSTRAP + McNEMAR (same holdout participants)
# ----------------------------------------------------------------------------
banner("7  PAIRED COMPARISONS vs. PRIMARY MODEL")
PM = ["AUROC", "Accuracy", "F1", "Recall", "Specificity"]
r = np.random.RandomState(RANDOM_STATE + 1); n = len(yt_h)
diff = {m: {k: [] for k in PM} for m in MODEL_ORDER if m != PRIMARY}
for _ in range(N_BOOT):
    i = r.randint(0, n, n)
    for m in diff:
        for k in PM:
            try: diff[m][k].append(MET_FN[k](yt_h[i], hold[m].y_prob.values[i]) -
                                   MET_FN[k](yt_h[i], hold[PRIMARY].y_prob.values[i]))
            except Exception: diff[m][k].append(np.nan)
paired = []
for m in diff:
    row = dict(Model=m, reference=PRIMARY)
    for k in PM:
        a = np.array(diff[m][k], float)
        row[k + "_diff"] = float(MET_FN[k](yt_h, hold[m].y_prob.values) - MET_FN[k](yt_h, hold[PRIMARY].y_prob.values))
        row[k + "_lo"], row[k + "_hi"] = float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    c1 = hold[m].y_pred.values == yt_h; c0 = hold[PRIMARY].y_pred.values == yt_h
    b, c = int((c1 & ~c0).sum()), int((~c1 & c0).sum())
    row["mcnemar_b_c"] = [b, c]
    row["mcnemar_p"] = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0
    paired.append(row)
pd.DataFrame(paired).to_csv(OUT / "table_paired_differences.csv", index=False)
VALUES["paired"] = paired
for p in paired:
    print(f"  {p['Model']:13s} dAUROC={p['AUROC_diff']:+.3f} [{p['AUROC_lo']:+.3f},{p['AUROC_hi']:+.3f}]  McNemar p={p['mcnemar_p']:.3f}")

# ----------------------------------------------------------------------------
# 8  REPEATED PARTICIPANT-LEVEL SPLITS (robustness to a single small holdout)
# ----------------------------------------------------------------------------
banner(f"8  {N_REPEATS} REPEATED STRATIFIED PARTICIPANT-LEVEL SPLITS")
rep = []
t0 = time()
for s in range(N_REPEATS):
    seed = 1000 + s; ts = make_split(seed)
    a = ~groups.isin(ts); b = groups.isin(ts)
    Xa, ya = X[a], y[a]; Xb, yb, gb = X[b], y[b], groups[b]
    mods = make_models(); pr = fit_predict(mods, Xa, ya, Xb, gb)
    for nme in MODEL_ORDER:
        d = subject_aggregate(gb.values, yb.values, pr[nme])
        m = metrics(d.y_true.values, d.y_pred.values, d.y_prob.values); m.update(Model=nme, seed=seed)
        rep.append(m)
    print(f"  split {s+1}/{N_REPEATS}  [{time()-t0:.0f}s]")
rep_df = pd.DataFrame(rep); rep_df.to_csv(OUT / "repeated_splits_all.csv", index=False)
rs = []
for nme in MODEL_ORDER:
    f = rep_df[rep_df.Model == nme]; row = {"Model": nme, "n_splits": len(f)}
    for k in ["Accuracy", "Recall", "Specificity", "F1", "AUROC"]:
        row[k + "_mean"] = f[k].mean(); row[k + "_sd"] = f[k].std()
        row[k + "_p2.5"] = f[k].quantile(.025); row[k + "_p97.5"] = f[k].quantile(.975)
    rs.append(row)
pd.DataFrame(rs).to_csv(OUT / "repeated_splits_summary.csv", index=False)
VALUES["repeated"] = rs

# ----------------------------------------------------------------------------
# 9  SEX ANALYSES
# ----------------------------------------------------------------------------
banner("9  SEX: STRATIFIED PERFORMANCE, PROXY CHECK, SEX-ADJUSTED SENSITIVITY")


def sex_table(res, tag):
    gmap = subj.set_index("subject_id")["sex"]
    out = []
    for nme, d in res.items():
        d = d.assign(sex=gmap.loc[d.group].values)
        for sv, lab in [(0, "Female"), (1, "Male")]:
            s = d[d.sex == sv]; yt, yb, yp = s.y_true.values, s.y_pred.values, s.y_prob.values
            tn, fp, fn, tp = confusion_matrix(yt, yb, labels=[0, 1]).ravel()
            try: au = roc_auc_score(yt, yp)
            except Exception: au = np.nan
            out.append(dict(set=tag, Model=nme, Sex=lab, n=len(s), n_pd=int(yt.sum()),
                            n_ctrl=int((1 - yt).sum()), TP=tp, FN=fn, FP=fp, TN=tn,
                            TPR=tp / (tp + fn) if tp + fn else np.nan,
                            TPR_lo=cp_ci(tp, tp + fn)[0], TPR_hi=cp_ci(tp, tp + fn)[1],
                            FPR=fp / (fp + tn) if fp + tn else np.nan,
                            FPR_lo=cp_ci(fp, fp + tn)[0], FPR_hi=cp_ci(fp, fp + tn)[1],
                            Accuracy=accuracy_score(yt, yb), AUROC=au))
    t = pd.DataFrame(out)
    # Fisher exact: male vs female FPR / TPR, per model
    for nme in t.Model.unique():
        f = t[(t.Model == nme) & (t.Sex == "Female")].iloc[0]; m = t[(t.Model == nme) & (t.Sex == "Male")].iloc[0]
        t.loc[t.Model == nme, "Fisher_p_FPR_gap"] = fisher_exact([[m.FP, m.TN], [f.FP, f.TN]])[1]
        t.loc[t.Model == nme, "Fisher_p_TPR_gap"] = fisher_exact([[m.TP, m.FN], [f.TP, f.FN]])[1]
    return t


sex_main = sex_table(hold, "main"); sex_main.to_csv(OUT / "S1_sex_stratified_main.csv", index=False)
VALUES["sex_main"] = sex_main.to_dict(orient="records")

# (b) can acoustic features alone predict sex?  (all participants, GroupKFold, XGBoost)
if gender_col:
    gk = GroupKFold(n_splits=N_SPLITS_CV); pr_s = np.zeros(len(df))
    for a, b in gk.split(X, sex, groups=groups):
        p = make_models()["XGBoost"]; p.fit(X.iloc[a], sex.iloc[a]); pr_s[b] = p.predict_proba(X.iloc[b])[:, 1]
    ds = pd.DataFrame({"g": groups.values, "t": sex.values, "p": pr_s}).groupby("g").agg(t=("t", "max"), p=("p", "mean"))
    au_sex = float(roc_auc_score(ds.t, ds.p))
    VALUES["sex_predictability_auroc"] = au_sex
    print(f"  Acoustic features -> sex, participant-level CV AUROC = {au_sex:.3f}")

# (c) sex-adjusted sensitivity: residualise inside each training fold
Xtr_s, Xte_s = with_sex(Xtr, str_), with_sex(Xte, ste)
adj = make_models(sex_adjust=True)
adj_prob = fit_predict(adj, Xtr_s, ytr, Xte_s, gte)
hold_adj = {n: subject_aggregate(gte.values, yte.values, adj_prob[n]) for n in MODEL_ORDER}
adj_rows = summarise(hold_adj, "sex_adjusted_holdout")
pd.DataFrame(adj_rows).to_csv(OUT / "table_holdout_sex_adjusted.csv", index=False)
sex_adj = sex_table(hold_adj, "sex_adjusted"); sex_adj.to_csv(OUT / "S1b_sex_stratified_sex_adjusted.csv", index=False)
VALUES["holdout_sex_adjusted"] = adj_rows; VALUES["sex_adjusted_strat"] = sex_adj.to_dict(orient="records")
for r_ in adj_rows:
    print(f"  [sex-adj] {r_['Model']:13s} AUROC={r_['AUROC']:.3f} Sens={r_['Recall']:.3f} Spec={r_['Specificity']:.3f}")

# ----------------------------------------------------------------------------
# 10  SHAP  (exact TreeExplainer on the CV-selected tree model)
# ----------------------------------------------------------------------------
banner("10  SHAP  (TreeExplainer; all holdout recordings; stability across folds)")
SHAP_MODEL = PRIMARY if PRIMARY in ("XGBoost", "RandomForest") else "XGBoost"
pipe = MODELS[SHAP_MODEL]


def family(f):
    if f.startswith("tqwt_"): return "TQWT (" + f.split("_")[1] + ")"
    if f.startswith("GNE"): return "GNE"
    if f.startswith("IMF"): return "IMF"
    if f.startswith("VFER"): return "VFER"
    if re.match(r"(mean|std)_MFCC", f): return "MFCC"
    if re.match(r"(mean|std)_\d+\w*_delta_delta|(mean|std)_delta_delta|(mean|std)_\d*\w*_?delta_delta", f): return "MFCC delta-delta"
    if re.match(r"(mean|std)_\d+\w*_delta|(mean|std)_0th_delta|(mean|std)_delta", f): return "MFCC delta"
    if "Log_energy" in f or "log_energy" in f: return "Log-energy"
    if re.match(r"app_|det_|Ea|Ed", f): return "Wavelet / entropy / TKEO"
    if re.match(r"f\d$", f): return "Formant frequency"
    if re.match(r"b\d$", f): return "Formant bandwidth"
    if re.search(r"Jitter|Shimmer", f): return "Jitter/Shimmer"
    if re.search(r"Intensity", f): return "Intensity"
    if re.search(r"Harmonicity", f): return "Harmonicity"
    if re.search(r"Pulse|Period", f): return "Pulse/period"
    if f in ("PPE", "DFA", "RPDE") or f.startswith("GQ"): return "Baseline / glottal quotient"
    return "Other"


def tree_shap(p, Xd):
    Xi = p.named_steps["imputer"].transform(Xd)
    ex = shap.TreeExplainer(p.named_steps["clf"])
    sv = ex.shap_values(Xi)
    if isinstance(sv, list): sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, -1]
    bv = ex.expected_value
    bv = float(np.ravel(bv)[-1])
    return Xi, sv, bv


Xi, sv, base = tree_shap(pipe, Xte)
mabs = np.abs(sv).mean(0)
order = np.argsort(-mabs)
# subject-level bootstrap CI for the top features
sub_idx = {g: np.where(gte.values == g)[0] for g in np.unique(gte.values)}
S = np.vstack([np.abs(sv[ix]).mean(0) for ix in sub_idx.values()])      # (n_subj, M)
rb = np.random.RandomState(RANDOM_STATE); bm = []
for _ in range(N_BOOT):
    bm.append(S[rb.randint(0, len(S), len(S))].mean(0))
bm = np.array(bm)
imp_rows = []
for rank, j in enumerate(order, 1):
    if rank <= TOP_K:
        rho = spearmanr(Xi[:, j], sv[:, j])[0] if np.std(Xi[:, j]) > 0 and np.std(sv[:, j]) > 0 else np.nan
        lo, hi = np.percentile(bm[:, j], [2.5, 97.5])
    else:
        rho = lo = hi = np.nan
    imp_rows.append(dict(rank=rank, feature=feature_cols[j], family=family(feature_cols[j]),
                         mean_abs_shap=mabs[j], ci_lo=lo, ci_hi=hi,
                         mean_signed_shap=sv[:, j].mean(), spearman_value_vs_shap=rho,
                         share_of_total=mabs[j] / mabs.sum()))
imp = pd.DataFrame(imp_rows)
# descriptive confounding check for the top features (all participants, participant-mean feature value):
# univariate AUROC for class and for sex (0.5 = no association; descriptive only, no model fitting)
_sm = df.groupby(id_col)[feature_cols].mean()
_cl = subj.set_index("subject_id").loc[_sm.index, "cls"].values
_sx = subj.set_index("subject_id").loc[_sm.index, "sex"].values
imp["auc_feature_vs_class"] = [roc_auc_score(_cl, _sm[f]) if r <= TOP_K else np.nan for f, r in zip(imp.feature, imp["rank"])]
imp["auc_feature_vs_sex"] = [roc_auc_score(_sx, _sm[f]) if r <= TOP_K else np.nan for f, r in zip(imp.feature, imp["rank"])]
imp.to_csv(OUT / "shap_feature_importance_v2.csv", index=False)
fam = (imp.groupby("family").agg(total_mean_abs_shap=("mean_abs_shap", "sum"),
                                 n_features=("feature", "count")).reset_index())
fam["share"] = fam.total_mean_abs_shap / fam.total_mean_abs_shap.sum()
fam = fam.sort_values("share", ascending=False); fam.to_csv(OUT / "shap_family_importance.csv", index=False)

# stability across CV folds (models refit on each training fold, explained on its validation fold)
fold_vec = []
for k, (a, b) in enumerate(folds):
    p = make_models()[SHAP_MODEL]; p.fit(Xtr.iloc[a], ytr.iloc[a])
    _, sv_k, _ = tree_shap(p, Xtr.iloc[b]); fold_vec.append(np.abs(sv_k).mean(0))
fold_vec = np.array(fold_vec)
tops = [set(np.argsort(-v)[:TOP_K]) for v in fold_vec]
jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(5) for j in range(i + 1, 5)]
rho_f = [spearmanr(fold_vec[i], fold_vec[j])[0] for i in range(5) for j in range(i + 1, 5)]
freq = np.zeros(len(feature_cols))
for t_ in tops:
    for j in t_: freq[j] += 1
hold_top = set(order[:TOP_K])
stab = dict(top_k=TOP_K, mean_pairwise_jaccard=float(np.mean(jac)),
            mean_pairwise_spearman=float(np.mean(rho_f)),
            n_features_in_top_k_in_ge3_folds=int((freq >= 3).sum()),
            n_holdout_top_k_in_ge3_folds=int(sum(freq[j] >= 3 for j in hold_top)),
            holdout_top_features_fold_frequency={feature_cols[j]: int(freq[j]) for j in order[:TOP_K]})
imp["fold_top20_frequency"] = [int(freq[feature_cols.index(f)]) for f in imp.feature]
imp.to_csv(OUT / "shap_feature_importance_v2.csv", index=False)
VALUES["shap"] = dict(model=SHAP_MODEL, output="log-odds (margin) of PD class",
                      n_explained_recordings=int(len(Xte)), n_explained_subjects=len(sub_idx),
                      background="TreeExplainer (tree_path_dependent; no background sample)",
                      top=imp.head(TOP_K).to_dict(orient="records"),
                      families=fam.head(10).to_dict(orient="records"),
                      stability=stab, n_zero_importance=int((mabs == 0).sum()))
print(f"  Model={SHAP_MODEL}; explained {len(Xte)} recordings; top-5:")
for _, r_ in imp.head(5).iterrows():
    print(f"    {r_['rank']}. {r_.feature:34s} {r_.mean_abs_shap:.4f}  rho={r_.spearman_value_vs_shap:+.2f} folds={r_.fold_top20_frequency}/5")
print(f"  Stability: Jaccard={stab['mean_pairwise_jaccard']:.2f}  Spearman={stab['mean_pairwise_spearman']:.2f}")

# figures: beeswarm + bar + local examples
plt.figure(figsize=(10, 9))
shap.summary_plot(sv, pd.DataFrame(Xi, columns=feature_cols), max_display=TOP_K, show=False)
plt.title(f"SHAP summary - {SHAP_MODEL} (all {len(Xte)} holdout recordings; log-odds)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "fig3_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(9, 7)); t = imp.head(TOP_K).iloc[::-1]
ax.barh(t.feature, t.mean_abs_shap, xerr=[t.mean_abs_shap - t.ci_lo, t.ci_hi - t.mean_abs_shap],
        color="#4C78A8", alpha=.9, capsize=2)
ax.set_xlabel("Mean |SHAP| (log-odds), 95% participant-bootstrap CI"); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig4_shap_bar.png", dpi=300, bbox_inches="tight"); plt.close()

pm = hold[SHAP_MODEL]
loc = {}
for lab, cond in [("TP", (pm.y_true == 1) & (pm.y_pred == 1)), ("TN", (pm.y_true == 0) & (pm.y_pred == 0)),
                  ("FP", (pm.y_true == 0) & (pm.y_pred == 1)), ("FN", (pm.y_true == 1) & (pm.y_pred == 0))]:
    c = pm[cond]
    if len(c) == 0: continue
    sid = c.iloc[(c.y_prob - c.y_prob.median()).abs().argsort().iloc[0]].group   # median-probability case
    pmean = float(pm[pm.group == sid].y_prob.iloc[0])                      # participant mean of 3 recording probabilities
    cand = sub_idx[sid]                                                     # waterfall explains ONE recording: the one closest to the participant mean
    ix = cand[np.argmin(np.abs(rec_prob[SHAP_MODEL][cand] - pmean))]
    prec = float(rec_prob[SHAP_MODEL][ix])                                  # recording-level probability = sigmoid(f(x))
    loc[lab] = dict(subject_id=float(sid), prob=pmean, recording_prob=prec, recording_logodds=float(base + sv[ix].sum()))
    try:
        shap.plots.waterfall(shap.Explanation(values=sv[ix], base_values=base, data=Xi[ix],
                                              feature_names=feature_cols), max_display=12, show=False)
        plt.title(f"{lab} participant id {int(sid)}: participant mean p(PD)={pmean:.2f}\nexplained recording: p(PD)={prec:.2f}, f(x)={base + sv[ix].sum():+.2f} (log-odds)", fontsize=9)
        plt.savefig(OUT / f"fig5_local_{lab}.png", dpi=250, bbox_inches="tight"); plt.close()
    except Exception as e:
        print("  waterfall failed:", e); plt.close()
VALUES["shap"]["local_examples"] = loc

# ----------------------------------------------------------------------------
# 11  OTHER FIGURES
# ----------------------------------------------------------------------------
banner("11  FIGURES")
fig, axs = plt.subplots(1, 5, figsize=(24, 4.5))
for ax, n in zip(axs, MODEL_ORDER):
    d = hold[n]; cm = confusion_matrix(d.y_true, d.y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred Ctrl", "Pred PD"], yticklabels=["True Ctrl", "True PD"],
                annot_kws={"size": 15, "weight": "bold"})
    m = metrics(d.y_true.values, d.y_pred.values, d.y_prob.values)
    ax.set_title(f"{n}\nSens={m['Recall']*100:.1f}%  Spec={m['Specificity']*100:.1f}%", fontsize=10, fontweight="bold")
plt.suptitle("Participant-level confusion matrices - holdout (n = %d)" % len(yt_h), fontweight="bold", y=1.04)
plt.tight_layout(); plt.savefig(OUT / "fig2_confusion_matrices.png", dpi=300, bbox_inches="tight"); plt.close()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
for n in MODEL_ORDER:
    d = hold[n]; f_, t_, _ = roc_curve(d.y_true, d.y_prob); p_, r_, _ = precision_recall_curve(d.y_true, d.y_prob)
    a1.plot(f_, t_, color=COLORS[n], lw=2, label=f"{n} (AUROC={roc_auc_score(d.y_true, d.y_prob):.3f})")
    a2.plot(r_, p_, color=COLORS[n], lw=2, label=f"{n} (AUPRC={average_precision_score(d.y_true, d.y_prob):.3f})")
a1.plot([0, 1], [0, 1], "k--", alpha=.4); a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - participant-level holdout")
a2.axhline(yt_h.mean(), color="k", ls="--", alpha=.4, label=f"Prevalence ({yt_h.mean():.2f})")
a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall - participant-level holdout")
a1.legend(); a2.legend(); a1.grid(alpha=.3); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig_roc_pr.png", dpi=300, bbox_inches="tight"); plt.close()

# participant flow figure
fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off")
ds_ = VALUES["dataset"]; sp_ = VALUES["split"]
boxes = [(.5, .88, f"PD Speech Features dataset\n{ds_['n_subjects']} participants ({ds_['n_pd']} PD, {ds_['n_ctrl']} controls)\n{ds_['n_recordings']} recordings (3 per participant)"),
         (.25, .52, f"Training partition\n{sp_['train']['n']} participants ({sp_['train']['pd']} PD / {sp_['train']['ctrl']} ctrl)\n{sp_['train_recordings']} recordings"),
         (.75, .52, f"Holdout partition (used once)\n{sp_['holdout']['n']} participants ({sp_['holdout']['pd']} PD / {sp_['holdout']['ctrl']} ctrl)\n{sp_['test_recordings']} recordings"),
         (.25, .15, f"5-fold GroupKFold CV\n(participant-grouped) -> model selection\nby mean CV AUROC = {PRIMARY}")]
for x_, y_, tx in boxes:
    ax.text(x_, y_, tx, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=.5", fc="#EAF1FB", ec="#4C78A8"))
for (x1, y1, x2, y2) in [(.5, .78, .25, .62), (.5, .78, .75, .62), (.25, .42, .25, .25)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))
ax.text(.5, .52, "no participant overlap", ha="center", fontsize=9, style="italic")
plt.savefig(OUT / "fig1_participant_flow.png", dpi=300, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# 12  SAVE
# ----------------------------------------------------------------------------
banner("12  SAVE VALUES + SOFTWARE")
VALUES["software"] = dict(python=sys.version.split()[0], sklearn=sklearn.__version__,
                          xgboost=xgboost.__version__, shap=shap.__version__,
                          numpy=np.__version__, pandas=pd.__version__)
VALUES["config"] = dict(seed=RANDOM_STATE, cv_folds=N_SPLITS_CV, n_bootstrap=N_BOOT,
                        n_repeats=N_REPEATS, threshold=THR, quick=bool(args.quick))
json.dump(VALUES, open(OUT / "manuscript_values.json", "w"), indent=1, default=fmt)
(OUT / "requirements_used.txt").write_text("\n".join(f"{k}=={v}" for k, v in VALUES["software"].items() if k != "python"))
print(json.dumps(VALUES["software"], indent=1))
print("\nDONE ->", OUT.resolve())
