#!/usr/bin/env python3
"""Builds the revised manuscript (.docx) and the response-to-reviewers (.docx)
directly from results_v2/ so every number in the text comes from the pipeline output.
Usage:  python build_manuscript.py results_v2 manuscript_revised.docx response_to_reviewers.docx"""
import sys, json, re
from pathlib import Path
import pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH

R = Path(sys.argv[1]); OUT_M = sys.argv[2]; OUT_R = sys.argv[3]
V = json.load(open(R / "manuscript_values.json"))
H = pd.read_csv(R / "table_holdout_subject_level.csv").set_index("Model")
CV = pd.read_csv(R / "cv_summary.csv").set_index("Model")
PO = pd.read_csv(R / "cv_pooled_oof_summary.csv").set_index("Model")
PD_ = pd.read_csv(R / "table_paired_differences.csv").set_index("Model")
RP = pd.read_csv(R / "repeated_splits_summary.csv").set_index("Model")
ADJ = pd.read_csv(R / "table_holdout_sex_adjusted.csv").set_index("Model")
SX = pd.read_csv(R / "S1_sex_stratified_main.csv")
SXA = pd.read_csv(R / "S1b_sex_stratified_sex_adjusted.csv")
IMP = pd.read_csv(R / "shap_feature_importance_v2.csv")
FAM = pd.read_csv(R / "shap_family_importance.csv")
D, SP, SH, SW = V["dataset"], V["split"], V["shap"], V["software"]
P = V["primary_model"]; ORDER = ["KNN", "XGBoost", "RandomForest", "SVM", "NaiveBayes"]
STAB = SH["stability"]


def p1(x): return f"{100*x:.1f}"
def ci(r, k): return f"{p1(r[k])} ({p1(r[k+'_lo'])}–{p1(r[k+'_hi'])})"
def d1(x): return f"{100*x:+.1f}"
def pv(p): return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------- sanity assertions: the narrative below depends on these facts ----------
assert P == "XGBoost"
assert H.loc[P, "AUROC"] == H["AUROC"].iloc[:5].max()
assert abs(H.loc["Majority class (all PD)", "Accuracy"] - H.loc["KNN", "Accuracy"]) < 1e-9
auc_excl0 = [m for m in PD_.index if PD_.loc[m, "AUROC_hi"] < 0 or PD_.loc[m, "AUROC_lo"] > 0]
assert auc_excl0 == ["RandomForest"], auc_excl0
spec_excl0 = [m for m in PD_.index if PD_.loc[m, "Specificity_hi"] < 0 or PD_.loc[m, "Specificity_lo"] > 0]
assert spec_excl0 == ["KNN"], spec_excl0
sens_excl0 = [m for m in PD_.index if PD_.loc[m, "Recall_hi"] < 0 or PD_.loc[m, "Recall_lo"] > 0]
assert sens_excl0 == ["NaiveBayes"], sens_excl0
for m in PD_.index:
    for k in ["Accuracy", "F1"]: assert PD_.loc[m, k+"_lo"] <= 1e-9 and PD_.loc[m, k+"_hi"] >= -1e-9, (m, k)
assert (PD_["mcnemar_p"] > 0.05).all()
assert (SX[SX.Sex == "Male"].set_index("Model").FPR > SX[SX.Sex == "Female"].set_index("Model").FPR).all()

# ---------- docx helpers ----------
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for s in doc.sections: s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(1)


def para(text, bold=False, italic=False, size=None, align=None, style=None):
    p = doc.add_paragraph(style=style)
    for seg in re.split(r"(\[\[.*?\]\])", text):
        if not seg: continue
        hl = seg.startswith("[[")
        r = p.add_run(seg[2:-2] if hl else seg); r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if hl: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    return p


def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)


def table(header, rows, caption, widths=None, size=8):
    para(caption, bold=True, size=10)
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for i, c in enumerate(header):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(size)
    for row in rows:
        cs = t.add_row().cells
        for i, c in enumerate(row):
            cs[i].text = ""; r = cs[i].paragraphs[0].add_run(str(c)); r.font.size = Pt(size)
    doc.add_paragraph()


def figure(fn, caption, w=6.3):
    f = R / fn
    if f.exists(): doc.add_picture(str(f), width=Inches(w))
    else: para(f"[[Figure file {fn} missing]]")
    para(caption, italic=True, size=10)


# =====================================================================================
# TITLE / ABSTRACT
# =====================================================================================
para("Classification of Parkinson's Disease Status From Sustained-Vowel Acoustic Features: A Participant-Level Machine Learning Benchmark",
     bold=True, size=15, align="c")
para("Dipraj Howlader¹, Tanvir Ahmed², Md. Mahbubur Rahman¹", align="c")
para("¹Department of Computer Science and Information Technology, Patuakhali Science and Technology University, Bangladesh; "
     "²Department of Computer Science and Engineering, University of Barishal, Bangladesh", size=9, align="c")
para("Correspondence: tahmed20.cse@bu.ac.bd", size=9, align="c")

h1("Abstract")
xg, svm_, rf, kn, nb = (H.loc[m] for m in ["XGBoost", "SVM", "RandomForest", "KNN", "NaiveBayes"])
para("Background. Many reports of voice-based Parkinson's disease (PD) classification evaluate recordings rather than people, so repeated recordings "
     "of one participant can fall on both sides of a train–test split. We built a participant-level benchmark on a public sustained-vowel dataset, "
     "quantified its uncertainty, examined sex as a possible confounder, and assessed how reliable post-hoc feature attributions are.")
para(f"Methods. The PD Speech Features dataset ({D['n_recordings']} recordings, {D['n_subjects']} participants: {D['n_pd']} PD, {D['n_ctrl']} controls; "
     f"{D['n_features']} acoustic features) was split into a stratified, participant-disjoint training partition ({SP['train']['n']} participants) and a holdout "
     f"({SP['holdout']['n']} participants: {SP['holdout']['pd']} PD, {SP['holdout']['ctrl']} controls). Five classifiers (KNN, XGBoost, random forest, SVM, naïve Bayes) "
     "were compared with imputation and scaling fitted inside each training fold. The model was selected by cross-validated AUROC on the training partition only, "
     "and the holdout was evaluated once. Metrics were computed per participant with bootstrap and exact confidence intervals, paired bootstrap model comparisons, "
     "majority-class and sex-only baselines, 20 repeated participant-level splits, sex-stratified and sex-adjusted analyses, and exact TreeExplainer SHAP with "
     "fold-wise stability assessment.")
para(f"Results. On the holdout, XGBoost had the highest AUROC ({ci(xg,'AUROC')}%), but only one pairwise AUROC difference excluded zero (vs. random forest). "
     f"Sensitivity was {p1(nb.Recall)}–{p1(svm_.Recall)}% across models whereas specificity was only {p1(kn.Specificity)}–{p1(nb.Specificity)}%; "
     f"accuracy of KNN ({p1(kn.Accuracy)}%) equalled the majority-class baseline ({p1(H.loc['Majority class (all PD)','Accuracy'])}%). "
     f"Controls were misclassified far more often among males than females in all five models, acoustic features predicted sex with AUROC {V['sex_predictability_auroc']:.2f}, "
     f"and removing sex-specific feature means lowered XGBoost AUROC to {p1(ADJ.loc['XGBoost','AUROC'])}%. "
     f"The top SHAP feature (std_delta_delta_log_energy) recurred in all five folds, but the overall ranking was only moderately stable "
     f"(mean pairwise Jaccard index of top-20 sets {STAB['mean_pairwise_jaccard']:.2f}).")
para("Conclusions. Under participant-level evaluation on one single-institution dataset, models detect PD participants with high sensitivity but low specificity, "
     "differences between models are mostly within sampling uncertainty, and sex is a plausible confounder that this dataset cannot resolve. "
     "The results are an internal benchmark and do not support claims about early detection, screening or clinical use.")
para("Keywords: Parkinson's disease; sustained vowel; acoustic features; participant-level evaluation; SHAP; confounding; machine learning benchmark", italic=True, size=10)

# =====================================================================================
# INTRODUCTION
# =====================================================================================
h1("1. Introduction")
para("Parkinson's disease (PD) is the second most prevalent neurodegenerative disorder; an estimated 11.77 million people were living with it in 2021, "
     "a 274% increase from 1990 [1], [5], and prevalence is projected to keep rising [2]. Motor signs usually become apparent after substantial "
     "dopaminergic loss, and non-motor features can precede the motor diagnosis [3], [4]. Voice impairment is common in PD, and dysphonia measures "
     "derived from sustained phonation have been related to symptom severity [6]. Because such measures can be extracted from short recordings, voice-based "
     "machine learning is an active research area.")
para("A recurrent methodological concern in this literature is the unit of evaluation. Participants in several public voice datasets contribute multiple "
     "recordings. If recordings, rather than participants, are randomly assigned to training and test sets, the same person's voice appears on both sides of the split, "
     "which can inflate performance estimates; fitting imputation or scaling on the full dataset before splitting is a second leakage route. Whether a given "
     "published figure suffers from these problems often cannot be determined from the report itself, so published figures are not directly comparable with participant-level estimates.")
para("Explainable-AI methods [7], [8] and in particular post-hoc attribution such as SHAP [10] are widely used to explain such classifiers, including in PD voice work [20], although inherently interpretable models are argued to be preferable for high-stakes decisions [9]. SHAP values describe how a feature contributes to one model's output on one dataset; "
     "they do not establish that a feature is a clinically validated biomarker [19]. Their numerical stability also depends on how they are computed and on the sample used.")
para("This study therefore (1) establishes a participant-level, leakage-controlled benchmark of five classifiers on a public dataset with explicit uncertainty quantification "
     "and pre-specified model selection; (2) reports sensitivity and specificity, and compares models on the same participants; (3) examines whether sex, which is unevenly "
     "distributed between cases and controls, influences the results; and (4) assesses the reliability of SHAP attributions across cross-validation folds. "
     "The study is an internal benchmark on a single dataset; it makes no claim about early detection, screening or clinical utility.")

# =====================================================================================
# RELATED WORK
# =====================================================================================
h1("2. Related work")
para("The 756-recording PD Speech Features corpus [21] is widely used. Reported accuracies range from 88.8% for KNN under an 80/20 split [11] to above 99% in "
     "reports that we could not verify as participant-grouped [13], [14]. Another study [12] used a smaller, different 195-instance dataset and noted that sample size limits generalisability. "
     "Reviews report that ensemble and distance-based classifiers tend to outperform simpler probabilistic models but that heterogeneous evaluation limits comparison [15]. "
     "Multimodal and spectrogram-based models have reported accuracies above 97% [16], [17], [18] on different data and, in some cases, unspecified protocols. "
     "Table 1 summarises what is and is not known about the evaluation unit of the studies most relevant to the present dataset. Because protocols differ and, in most cases, "
     "the evaluation unit is not stated, we make no superiority claim over any of them.")
table(["Study", "Data", "Reported evaluation (as described)", "Reported performance", "Participant-grouped?", "Comparable to this study?"],
      [["Soni et al. [11]", "756-recording corpus [21]", "Random 80/20 split; KNN, GB, DT", "Acc 88.8% (KNN)", "Not stated [[verify in source]]", "No"],
       ["Bashir et al. [12]", "195-instance dataset (different)", "RFE + XGBoost", "Acc 95%", "Not applicable (different dataset)", "No"],
       ["Rhythm et al. [13]", "[[verify dataset]]", "KNN", "Acc 96.61%", "Not stated [[verify in source]]", "No"],
       ["Shibina & Thasleema [14]", "Review paper citing a RF study", "Leave-one-out CV (as summarised)", "Acc 99.94%", "Not stated [[replace by primary source]]", "No"],
       ["This study", "756-recording corpus [21]", "Participant-disjoint holdout + grouped CV", "See Results", "Yes", "—"]],
      "Table 1. Evaluation protocols of selected related studies. Entries marked in yellow must be checked against the original papers before submission.")

# =====================================================================================
# METHODS
# =====================================================================================
h1("3. Methods")
h2("3.1 Dataset")
para(f"We analysed the public PD Speech Features dataset [21] (UCI Machine Learning Repository, DOI 10.24432/C5T60W; Department of Neurology, Cerrahpaşa Faculty of Medicine, Istanbul University). "
     f"It contains {D['n_recordings']} recordings of a sustained vowel, three per participant, from {D['n_subjects']} participants: {D['n_pd']} with PD "
     f"({D['pd_male']} male, {D['pd_female']} female; age 65.1 ± 10.9 years, range 33–87) and {D['n_ctrl']} healthy controls ({D['ctrl_male']} male, {D['ctrl_female']} female; "
     f"age 61.1 ± 8.9 years, range 41–82), as described by the data providers [21]. The public file does not contain age, disease stage, medication status or the diagnostic criteria; "
     "these cannot be analysed here and the reader is referred to [21]. PD prevalence differs by sex: "
     f"{100*D['ctrl_male']/(D['ctrl_male']+D['pd_male']):.1f}% of male but {100*D['ctrl_female']/(D['ctrl_female']+D['pd_female']):.1f}% of female participants are controls.")
para(f"The file we analysed has {D['n_columns_total']} columns: the participant identifier (id), sex (gender; 0 = female, 1 = male), the class label (0 = control, 1 = PD) and {D['n_features']} acoustic features "
     "(baseline, intensity, formant, bandwidth, vocal-fold, MFCC, wavelet and TQWT features; recurrence and fractal measures such as RPDE and DFA follow [22]). "
     "The id, gender and class columns were never used as model inputs (the sex column was used only for the sex analyses described below). "
     "Reports of 754 or 755 features differ in which of these columns are counted; the complete list of the 752 input columns is provided (S4 Table). "
     f"The file contains {D['missing_cells']} missing values, so no imputation was actually required (the imputer is retained in every pipeline as a safeguard); "
     f"{D['constant_features']} features were constant.")

h2("3.2 Participant-level split and flow")
para(f"Participants were split once into a training partition and a holdout partition using stratified random assignment within each class (seed 42; 20% of each class to holdout). "
     f"All three recordings of a participant stay together. Participant-ID overlap between partitions was verified to be zero. Table 2 gives exact counts; Fig. 1 shows the flow. "
     "The participant-level assignment is provided (S5 Table).")
trn, hol = SP["train"], SP["holdout"]
table(["Partition", "Participants", "Recordings", "PD", "Controls", "PD male/female", "Controls male/female"],
      [["Training", trn["n"], SP["train_recordings"], trn["pd"], trn["ctrl"], f"{trn['pd_male']}/{trn['pd_female']}", f"{trn['ctrl_male']}/{trn['ctrl_female']}"],
       ["Holdout (evaluated once)", hol["n"], SP["test_recordings"], hol["pd"], hol["ctrl"], f"{hol['pd_male']}/{hol['pd_female']}", f"{hol['ctrl_male']}/{hol['ctrl_female']}"],
       ["Total", D["n_subjects"], D["n_recordings"], D["n_pd"], D["n_ctrl"], f"{D['pd_male']}/{D['pd_female']}", f"{D['ctrl_male']}/{D['ctrl_female']}"]],
      "Table 2. Participant and recording counts by partition.")
figure("fig1_participant_flow.png", "Fig. 1. Participant flow and data partitioning.", 5.5)

h2("3.3 Classifiers, preprocessing and model selection")
para("Five classifiers were compared (Table 3). Every model is a scikit-learn [28] pipeline in which median imputation (and, for KNN, SVM and naïve Bayes, z-score standardisation) is fitted only on the training "
     "data of the relevant fit and then applied to validation or holdout data. Tree ensembles do not use scaling. No hyperparameters were tuned and the decision threshold was fixed a priori at 0.5; "
     "the holdout therefore informed no setting. Model selection was pre-specified as the highest mean AUROC in 5-fold participant-grouped (GroupKFold) cross-validation within the training partition "
     f"(selected model: {P}). The holdout was evaluated once for all five models; all five are reported.")
table(["Model", "Pipeline", "Settings (all other arguments scikit-learn / XGBoost defaults)"],
      [["KNN [23]", "Imputer → StandardScaler → KNN", "k = 5, Euclidean distance"],
       ["XGBoost [27]", "Imputer → XGBClassifier", "n_estimators = 100, max_depth = 3, learning_rate = 0.2, eval_metric = logloss, random_state = 42"],
       ["Random forest [24]", "Imputer → RandomForest", "n_estimators = 100, random_state = 42"],
       ["SVM [25]", "Imputer → StandardScaler → SVC", "RBF kernel, probability = True (Platt scaling), random_state = 42"],
       ["Naïve Bayes [26]", "Imputer → StandardScaler → GaussianNB", "defaults"]],
      "Table 3. Model specifications.")

h2("3.4 Participant-level aggregation and metrics")
para("Recording-level positive-class probabilities were averaged over each participant's three recordings and thresholded at 0.5; all metrics are computed per participant. We report accuracy, precision, "
     "sensitivity (recall), specificity, F1, AUROC and AUPRC with confusion-matrix counts. 95% CIs for accuracy, precision, sensitivity, F1, AUROC and AUPRC use a non-parametric bootstrap over participants "
     f"({V['config']['n_bootstrap']} resamples, seed 42); specificity CIs are exact Clopper–Pearson intervals. Because the holdout contains only {hol['ctrl']} controls, specificity intervals are wide. "
     "Models were compared on the same holdout participants with a paired bootstrap (differences relative to the selected model, 95% percentile intervals) and exact McNemar tests on participant-level correctness. "
     "As reference points we report a majority-class classifier (predict PD for everybody) and a sex-only logistic regression trained on the training partition. To use all training participants for estimation, "
     "pooled out-of-fold participant-level predictions from the five CV folds (201 participants) were also evaluated. Finally, to show how much the single holdout matters, the entire protocol "
     f"(fit on training, evaluate on holdout) was repeated on {V['config']['n_repeats']} further stratified participant-level random splits (seeds 1000–{1000+V['config']['n_repeats']-1}); these results are descriptive because training sets overlap.")

h2("3.5 Sex analyses")
para("(i) Sex-stratified holdout metrics are reported with exact CIs and Fisher exact tests for the male–female difference in false-positive and true-positive rates. "
     "(ii) To test whether acoustic features encode sex, XGBoost was trained to predict sex from the acoustic features in participant-grouped 5-fold CV over all participants. "
     "(iii) As a sensitivity analysis, the sex-specific mean of each feature was estimated on the training data only and subtracted inside each pipeline (equivalent to residuals of a feature-on-sex regression), "
     "and all models were refitted. This is not a causal correction: because PD prevalence differs by sex, it may also remove disease-related variance. Age could not be examined because it is not in the public file.")

h2("3.6 SHAP analysis")
para(f"SHAP values were computed with the exact TreeExplainer ({SW['shap']}) for the selected tree model ({SH['model']}), in the original (imputed) feature space and in log-odds units of the PD class. "
     f"All {SH['n_explained_recordings']} holdout recordings ({SH['n_explained_subjects']} participants) were explained; each point in Fig. 3 is one recording. TreeExplainer needs no background sample and no random permutation budget. "
     "Mean absolute SHAP values were summarised with participant-level bootstrap CIs; the direction of association was described by the Spearman correlation between a feature's value and its SHAP value. "
     "Stability was assessed by refitting the model on each CV training fold, explaining the corresponding validation fold, and comparing the top-20 feature sets (pairwise Jaccard index) and full rankings (Spearman correlation) across folds. "
     "As a descriptive check for confounding, the univariate AUROC of each top feature (participant-mean value) for class and for sex is reported. "
     "Attributions describe the model, not physiology. Local explanations are shown for one holdout participant of each outcome type (TP, TN, FP, FN). A waterfall plot explains a single recording: we chose the recording whose probability was closest to the participant mean; f(x) in each plot is that recording's log-odds, and the plot title also gives the participant-level mean probability used for classification. Because the logistic function is non-linear, the two can lie on opposite sides of 0.5 (f(x) = 0).")

h2("3.7 Software, reproducibility and reporting")
para(f"Python {SW['python']}, scikit-learn {SW['sklearn']}, XGBoost {SW['xgboost']}, SHAP {SW['shap']}, NumPy {SW['numpy']}, pandas {SW['pandas']}; random seed 42 throughout. "
     "The analysis script, participant-level split, holdout predictions and all result tables are provided (see Code and Data Availability). Reporting follows the TRIPOD+AI guideline [29] "
     "[[S3 File: completed TRIPOD+AI checklist – to be added by the authors]].")

# =====================================================================================
# RESULTS
# =====================================================================================
h1("4. Results")
h2("4.1 Cross-validation on the training partition")
rows = []
for m in ORDER:
    c = CV.loc[m]; pr = PO.loc[m]
    rows.append([m] + [f"{c[k+'_mean']:.3f} ± {c[k+'_sd']:.3f}" for k in ["Accuracy", "Recall", "Specificity", "AUROC"]] + [f"{pr.AUROC:.3f} ({pr.AUROC_lo:.3f}–{pr.AUROC_hi:.3f})"])
table(["Model", "Accuracy", "Sensitivity", "Specificity", "AUROC (fold mean ± SD)", "AUROC pooled out-of-fold, n = 201 (95% CI)"], rows,
      "Table 4. Five-fold participant-grouped cross-validation within the training partition (about 40 participants per fold). Per-fold values: S2 Table.")
para(f"XGBoost had the highest mean CV AUROC ({CV.loc['XGBoost','AUROC_mean']:.3f} ± {CV.loc['XGBoost','AUROC_sd']:.3f}) and was therefore the pre-specified primary model. "
     "Fold-to-fold variation was large (AUROC SD 0.10–0.17), reflecting about 40 participants per validation fold. In pooled out-of-fold predictions "
     f"XGBoost had AUROC {PO.loc['XGBoost','AUROC']:.3f} ({PO.loc['XGBoost','AUROC_lo']:.3f}–{PO.loc['XGBoost','AUROC_hi']:.3f}); "
     f"sensitivity was high ({p1(PO['Recall'].min())}–{p1(PO['Recall'].max())}%) whereas specificity was {p1(PO['Specificity'].min())}–{p1(PO['Specificity'].max())}%.")

h2("4.2 Holdout performance (participant level)")
rows = []
for m in ORDER + ["Majority class (all PD)", "Sex-only logistic baseline"]:
    r = H.loc[m]
    if m in ORDER:
        rows.append([m, ci(r, "Accuracy"), ci(r, "Recall"), ci(r, "Specificity") if False else f"{p1(r.Specificity)} ({p1(r.Specificity_lo)}–{p1(r.Specificity_hi)})",
                     ci(r, "Precision"), ci(r, "F1"), ci(r, "AUROC"), ci(r, "AUPRC"), f"{int(r.TP)}/{int(r.FP)}/{int(r.TN)}/{int(r.FN)}"])
    else:
        a = f"{p1(r.Accuracy)}" + (f" ({p1(r.Accuracy_lo)}–{p1(r.Accuracy_hi)})" if not np.isnan(r.Accuracy_lo) else "")
        au = f"{p1(r.AUROC)}" + (f" ({p1(r.AUROC_lo)}–{p1(r.AUROC_hi)})" if not np.isnan(r.AUROC_lo) else "")
        rows.append([m, a, p1(r.Recall), p1(r.Specificity), p1(r.Precision), p1(r.F1), au, p1(r.AUPRC), f"{int(r.TP)}/{int(r.FP)}/{int(r.TN)}/{int(r.FN)}"])
table(["Model", "Accuracy %", "Sensitivity %", "Specificity %", "Precision %", "F1 %", "AUROC %", "AUPRC %", "TP/FP/TN/FN"], rows,
      f"Table 5. Participant-level holdout performance (n = {hol['n']}: {hol['pd']} PD, {hol['ctrl']} controls). Values are % (95% CI). The holdout was evaluated once; model selection used the training partition only.", size=7)
para(f"XGBoost had the highest AUROC ({ci(xg,'AUROC')}%), followed by KNN, SVM, random forest and naïve Bayes (75.7–78.0%, wide overlapping CIs). SVM had the highest accuracy ({p1(svm_.Accuracy)}%) and F1 ({p1(svm_.F1)}%); "
     f"SVM and random forest had the highest sensitivity (each {p1(svm_.Recall)}%; one missed PD participant). Specificity was low for every model ({p1(kn.Specificity)}% for KNN, {p1(xg.Specificity)}% for XGBoost and random forest, "
     f"{p1(svm_.Specificity)}% for SVM, {p1(nb.Specificity)}% for naïve Bayes; Table 5), i.e. between 5 and 11 of the 13 control participants were classified as PD, depending on the model. The accuracy of KNN equals the majority-class baseline ({p1(kn.Accuracy)}%), and "
     f"the sex-only baseline gave AUROC {p1(H.loc['Sex-only logistic baseline','AUROC'])}% and also classified everyone as PD. High recall therefore partly reflects the {100*D['prevalence']:.1f}% PD prevalence of this cohort. "
     "Participant-level confusion matrices are in Fig. 2 and ROC/precision–recall curves in Fig. 6.")
figure("fig2_confusion_matrices.png", "Fig. 2. Participant-level confusion matrices on the holdout set.")

h2("4.3 Paired comparison of models")
rows = []
for m in PD_.index:
    r = PD_.loc[m]
    f = lambda k: f"{d1(r[k+'_diff'])} ({d1(r[k+'_lo'])} to {d1(r[k+'_hi'])})"
    rows.append([m, f("AUROC"), f("Accuracy"), f("F1"), f("Recall"), f("Specificity"), f"{r['mcnemar_p']:.2f}"])
table(["Model vs. XGBoost", "Δ AUROC", "Δ Accuracy", "Δ F1", "Δ Sensitivity", "Δ Specificity", "McNemar p (correctness)"], rows,
      "Table 6. Differences (percentage points) relative to XGBoost on the same 51 holdout participants; paired bootstrap 95% intervals.")
nbr, knr, rfr = PD_.loc["NaiveBayes"], PD_.loc["KNN"], PD_.loc["RandomForest"]
para(f"Three differences relative to XGBoost had 95% intervals that excluded zero: AUROC for random forest ({d1(rfr.AUROC_diff)}, {d1(rfr.AUROC_lo)} to {d1(rfr.AUROC_hi)}; only just), "
     f"specificity for KNN ({d1(knr.Specificity_diff)}, {d1(knr.Specificity_lo)} to {d1(knr.Specificity_hi)}) and sensitivity for na\u00efve Bayes ({d1(nbr.Recall_diff)}, {d1(nbr.Recall_lo)} to {d1(nbr.Recall_hi)}). "
     "All other intervals, including every accuracy and F1 difference, included or touched zero, and no McNemar test was significant. These intervals are unadjusted for multiple comparisons. "
     "The data thus do not allow a confident overall ranking of the five classifiers on the holdout.")
h2("4.4 Robustness to the choice of holdout")
rows = [[m] + [f"{p1(RP.loc[m,k+'_mean'])} ± {p1(RP.loc[m,k+'_sd'])}" for k in ["AUROC", "Accuracy", "Recall", "Specificity"]] for m in ORDER]
table(["Model", "AUROC %", "Accuracy %", "Sensitivity %", "Specificity %"], rows,
      f"Table 7. Mean ± SD over {V['config']['n_repeats']} repeated stratified participant-level splits (descriptive; training sets overlap).")
para(f"Across {V['config']['n_repeats']} repeated splits XGBoost again had the highest mean AUROC ({p1(RP.loc['XGBoost','AUROC_mean'])} ± {p1(RP.loc['XGBoost','AUROC_sd'])}%), "
     f"with specificity ({p1(RP.loc['XGBoost','Specificity_mean'])}%) well below sensitivity ({p1(RP.loc['XGBoost','Recall_mean'])}%). The split-to-split SD of AUROC (4–6 percentage points) shows how much a single 51-participant holdout can vary.")
figure("fig_roc_pr.png", "Fig. 6. ROC and precision–recall curves on the holdout set (participant level).")

h2("4.5 Sex: stratified performance, proxy check and sex-adjusted sensitivity analysis")
s = SX.set_index(["Model", "Sex"])
rows = []
for m in ORDER:
    f, ml = s.loc[(m, "Female")], s.loc[(m, "Male")]
    rows.append([m, f"{int(f.TP)}/{int(f.TP+f.FN)}", f"{int(ml.TP)}/{int(ml.TP+ml.FN)}",
                 f"{int(f.FP)}/{int(f.FP+f.TN)} ({p1(f.FPR_lo)}–{p1(f.FPR_hi)})", f"{int(ml.FP)}/{int(ml.FP+ml.TN)} ({p1(ml.FPR_lo)}–{p1(ml.FPR_hi)})", pv(f.Fisher_p_FPR_gap)])
table(["Model", "PD detected – female", "PD detected – male", "Controls misclassified – female (exact 95% CI of FPR %)", "Controls misclassified – male (exact 95% CI of FPR %)", "Fisher p (FPR, male vs female)"],
      rows, "Table 8. Sex-stratified holdout results (female: 10 PD, 5 controls; male: 28 PD, 8 controls). Fisher p-values are unadjusted and the five models share the same controls. All metrics: S1 Table.")
para(f"In all five models a larger fraction of male than female controls was classified as PD (male false-positive rate 62.5–100%, female 0–60%). For XGBoost, 7 of 8 male but 0 of 5 female controls were misclassified (Fisher p = {pv(s.loc[('XGBoost','Female'),'Fisher_p_FPR_gap'])}). "
     "Per-sex AUROCs rest on 5 and 8 controls and are not interpretable as stable estimates. "
     f"Acoustic features alone predicted sex with a participant-level cross-validated AUROC of {V['sex_predictability_auroc']:.2f}, so the features carry strong sex information. "
     "In the sex-adjusted sensitivity analysis (Table 9) XGBoost AUROC fell from "
     f"{p1(xg.AUROC)}% to {ci(ADJ.loc['XGBoost'],'AUROC')}%, specificity remained low, and the male–female difference in false-positive rate persisted (XGBoost: "
     f"{int(SXA[(SXA.Model=='XGBoost')&(SXA.Sex=='Male')].FP.iloc[0])}/8 male vs {int(SXA[(SXA.Model=='XGBoost')&(SXA.Sex=='Female')].FP.iloc[0])}/5 female controls). "
     "With 13 controls these data cannot establish whether this pattern reflects sex-related acoustic differences, an interaction of sex with disease status, or chance; the consistent direction across models is notable, although the models share the same 13 controls and are therefore not independent evidence.")
rows = [[m, ci(ADJ.loc[m], "AUROC"), p1(ADJ.loc[m, "Recall"]), f"{p1(ADJ.loc[m,'Specificity'])} ({p1(ADJ.loc[m,'Specificity_lo'])}–{p1(ADJ.loc[m,'Specificity_hi'])})", p1(ADJ.loc[m, "Accuracy"])] for m in ORDER]
table(["Model", "AUROC % (95% CI)", "Sensitivity %", "Specificity % (exact 95% CI)", "Accuracy %"], rows,
      "Table 9. Sensitivity analysis: sex-specific feature means removed inside each training pipeline (holdout, participant level).")

h2("4.6 SHAP analysis of the selected model")
t10 = IMP.head(10)
dirtxt = lambda r: ("higher value → higher PD log-odds" if r > 0.3 else ("higher value → lower PD log-odds" if r < -0.3 else "mixed / non-monotonic"))
rows = [[int(r["rank"]), r.feature, r.family, f"{r.mean_abs_shap:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", f"{r.spearman_value_vs_shap:+.2f}", dirtxt(r.spearman_value_vs_shap),
         f"{int(r.fold_top20_frequency)}/5", f"{r.auc_feature_vs_class:.2f}", f"{r.auc_feature_vs_sex:.2f}"] for _, r in t10.iterrows()]
table(["Rank", "Feature", "Family", "Mean |SHAP| (95% CI)", "ρ (value vs SHAP)", "Direction", "Top-20 in CV folds", "Univariate AUROC vs class", "Univariate AUROC vs sex"], rows,
      f"Table 10. Top-10 features by mean |SHAP| for {SH['model']} on all {SH['n_explained_recordings']} holdout recordings (log-odds units). Univariate AUROCs use participant-mean feature values over all participants and are descriptive.", size=7)
top1 = IMP.iloc[0]
para(f"{SH['n_zero_importance']} of {D['n_features']} features had zero attribution (never used by the trees). The highest-ranked feature was {top1.feature} (mean |SHAP| {top1.mean_abs_shap:.3f}, 95% CI {top1.ci_lo:.3f}–{top1.ci_hi:.3f}); "
     f"higher values were associated with higher PD log-odds (ρ = {top1.spearman_value_vs_shap:+.2f}) and the feature was among the top 20 in all five CV folds; its univariate AUROC was {top1.auc_feature_vs_class:.2f} for class and {top1.auc_feature_vs_sex:.2f} for sex. "
     "The second-ranked feature (tqwt_entropy_log_dec_27) was more strongly associated with sex (univariate AUROC "
     f"{IMP.iloc[1].auc_feature_vs_sex:.2f}) than with class ({IMP.iloc[1].auc_feature_vs_class:.2f}), so it may act partly as a sex proxy. By feature family, MFCC delta-delta features ({100*FAM.iloc[0].share:.1f}% of total mean |SHAP|) and TQWT entropy features "
     f"({100*FAM.iloc[1].share:.1f}%) contributed most (S7 Table), i.e. the explanation is spread across many correlated features rather than a few.")
para(f"Stability across folds was only moderate: the mean pairwise Jaccard index of top-20 sets was {STAB['mean_pairwise_jaccard']:.2f} and the mean pairwise Spearman correlation of full rankings {STAB['mean_pairwise_spearman']:.2f}; "
     f"{STAB['n_holdout_top_k_in_ge3_folds']} of the 20 holdout top features were in the top 20 of at least three of five folds. Only the first few features should therefore be treated as reasonably stable, and the order within the list is not reliable. "
     "Figs. 3–5 show the global and local explanations.")
figure("fig3_shap_beeswarm.png", "Fig. 3. SHAP summary (beeswarm) for XGBoost on all 153 holdout recordings; each point is one recording.", 5.6)
figure("fig4_shap_bar.png", "Fig. 4. Top-20 features by mean |SHAP| with participant-bootstrap 95% CIs.", 5.2)
para("Fig. 5. Local explanations (waterfall plots) for one holdout participant (median-probability case) of each outcome type: true positive, true negative, false positive, false negative. Each plot explains one recording of that participant (the one closest to the participant mean); f(x) is that recording's log-odds, and p(PD) in the title is the participant-level mean of the three recording probabilities, which determines the class. The two can fall on opposite sides of the threshold because the logistic function is non-linear.", italic=True, size=10)
for lab in ["TP", "TN", "FP", "FN"]:
    f = R / f"fig5_local_{lab}.png"
    if f.exists(): doc.add_picture(str(f), width=Inches(4.6)); para(f"Fig. 5{'abcd'['TP TN FP FN'.split().index(lab)]}. {lab} participant.", italic=True, size=9)

# =====================================================================================
# DISCUSSION
# =====================================================================================
h1("5. Discussion")
para("Under a participant-disjoint protocol with preprocessing inside the pipelines, the five classifiers detected PD participants in this cohort with high sensitivity, but specificity was low (15–62% on the holdout; 35–67% in pooled cross-validation). "
     "Because 74.6% of participants have PD, accuracy of 75–86% is close to the 74.5% obtained by labelling everyone as PD, and high sensitivity must be read together with specificity. "
     "Precision also cannot be transferred to other populations: with XGBoost's holdout sensitivity and specificity "
     f"({p1(xg.Recall)}% and {p1(xg.Specificity)}%), the positive predictive value at an illustrative 1% prevalence would be about "
     f"{100*(xg.Recall*0.01)/(xg.Recall*0.01+(1-xg.Specificity)*0.99):.1f}%. This is an arithmetic illustration, not an estimate for any real setting.")
para("XGBoost was selected by cross-validation on the training data and had the highest holdout AUROC and the highest mean AUROC over 20 repeated splits, consistent with the usual strength of boosted trees on tabular data [27]. "
     "However, with 51 holdout participants the paired intervals show that most differences between classifiers cannot be distinguished from sampling variation, so we do not declare a winner; "
     "SVM and random forest had slightly higher accuracy and sensitivity but also overlapping intervals. We did not compare with the recording-level figures in Table 1 by re-running recording-level splits ourselves, and we therefore do not quantify how much "
     "those protocols inflate performance; we only note that their figures are not comparable to participant-level estimates.")
para("Sex deserves particular attention. Cases and controls differ in sex composition, acoustic features predict sex with AUROC 0.96, and control participants were misclassified more often among males in every model, a pattern that persisted after removing sex-specific feature means. "
     "This is consistent with sex-related or sex-by-disease-related acoustic cues being used by the classifiers, but 13 holdout controls (8 male, 5 female) cannot settle it, and age and recording-related factors could not be examined. "
     "We therefore report the sex analysis as a warning about confounding, not as a fairness assessment: evidence on fairness would require balanced and larger samples from the intended population.")
para("SHAP indicates which features the selected model relies on; it does not show that a feature is a biomarker. The feature that ranked first, std_delta_delta_log_energy, is a statistic of the second temporal derivative of log-energy "
     "in short frames; it was stable across folds and individually associated with class, but its physiological meaning has not been established by this study. Some other highly ranked features were at least as strongly associated with sex as with class. "
     "The moderate stability of the ranking across folds, the spread of attribution across correlated feature families and the fact that 507 of 752 features were never used mean that the SHAP output should be viewed as a description of this model on this dataset. "
     "Prior literature relating dysphonia measures to PD [6] provides context for why voice features may be informative, but we do not interpret individual features physiologically.")
h2("5.1 Limitations")
para("(1) Single public dataset from one institution, one language and one vowel, with three recordings per participant; no external validation. "
     "(2) The holdout has 51 participants and only 13 controls; confidence intervals are wide and specificity is poorly determined. "
     "(3) Age, disease stage, medication status and diagnostic criteria are not available in the public file; age differs between groups in the source description [21] and could not be adjusted. "
     "(4) Sex is a probable confounder that could be probed but not resolved; the sex-adjusted analysis is a sensitivity analysis only. "
     "(5) No hyperparameters were tuned and the threshold was fixed at 0.5, so results describe default settings; tuning in nested cross-validation might change the model ranking. "
     "(6) Performance in a 3:1 case-control cohort does not describe performance in a population with lower prevalence. "
     "(7) Attributions describe model behaviour, are moderately unstable and cannot establish biomarker validity. "
     "(8) Repeated-split results are descriptive because training sets overlap.")

h1("6. Conclusion")
para("This participant-level benchmark shows that, on a public single-institution sustained-vowel dataset, standard classifiers reach high sensitivity but low specificity, "
     "that differences between them are mostly within sampling uncertainty, that sex is a plausible confounder, and that SHAP attributions are only moderately stable. "
     "The results are an internal benchmark and do not support claims about early detection, community screening, differential diagnosis, fairness or clinical use; these would require external, prospective validation in the relevant populations.")

h1("Supporting information")
para("S1 Table. Sex-stratified metrics, all models (S1_sex_stratified_main.csv; sex-adjusted: S1b). • S2 Table. Per-fold cross-validation metrics (S2_cv_per_fold_all_models.csv). • S3 File. TRIPOD+AI checklist [[to be added]]. "
     "• S4 Table. The 752 input feature names (feature_columns_used.csv). • S5 Table. Participant-level partition assignment (split_assignment_participants.csv). "
     "• S6 Table. Holdout recording- and participant-level predictions. • S7 Table. Full SHAP importance and family importance (shap_feature_importance_v2.csv, shap_family_importance.csv).")
h1("Code availability")
para("All analysis code (pd_revised_analysis_v2.py), result tables and the manuscript build script are available at [[GitHub/Zenodo URL – insert before submission]]. Running the script on the UCI file reproduces every table and figure in this paper.")
h1("Ethics statement")
para("This study is a secondary analysis of a publicly available, de-identified dataset (UCI Machine Learning Repository, DOI 10.24432/C5T60W). No new participants were recruited. Ethical approval and consent for the original data collection are described by the data providers [21].")
h1("Data availability statement")
para("The PD Speech Features dataset is publicly available from the UCI Machine Learning Repository (DOI 10.24432/C5T60W) [21]. The participant-level split, holdout predictions, feature list and all derived result files generated in this study are available at the repository given under Code availability [[insert URL]].")

h1("References")
REFS = open(Path(__file__).with_name("references.txt")).read().strip().split("\n")
for r_ in REFS: para(r_, size=9)
doc.save(OUT_M)

# =====================================================================================
# RESPONSE TO REVIEWERS
# =====================================================================================
doc = Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
para("Response to Reviewers", bold=True, size=15)
para("We thank both reviewers. We re-ran the whole analysis (script pd_revised_analysis_v2.py) and rewrote the manuscript. Section numbers refer to the revised manuscript. "
     "Several results changed in tone: the participant-level analysis does not support a clear model ranking, and specificity is low.")
resp = [
 ("R1-1  Evaluation unit; verify holdout", f"Table 5 is now participant-level (n = {hol['n']}: {hol['pd']} PD, {hol['ctrl']} controls; TP+FN = 38, FP+TN = 13 for every model). Exact counts by class and sex for both partitions are in Table 2 and Fig. 1; zero participant-ID overlap was asserted in code; the participant assignment, recording-level and participant-level holdout predictions are provided (S5, S6) so that Table 5 can be regenerated."),
 ("R1-2  Model selection vs final evaluation", "No hyperparameters were tuned and the threshold was fixed at 0.5. The primary model is now selected by mean CV AUROC on the training partition only (XGBoost), and the holdout is evaluated once (Section 3.3). Pooled out-of-fold participant-level metrics (n = 201) are added (Table 4). We did not implement nested CV because no hyperparameters are tuned; this is listed as a limitation. Seeds, folds and fold composition: Sections 3.2–3.4, S2."),
 ("R1-3  Uncertainty, precision, recall ranking", "Bootstrap CIs for all metrics, exact CIs for specificity, and sensitivity AND specificity are reported (Table 5). Models are compared on the same participants with a paired bootstrap and McNemar tests (Table 6); we no longer declare a winner. The recall ranking was corrected. The discussion states that precision depends on prevalence (74.6% here) and gives an illustrative PPV at 1% prevalence."),
 ("R1-4  Cohort composition and predictors", f"The exact 752-column input list is provided (S4); id, gender and class handling is stated (Section 3.1). The actual number of missing values is {D['missing_cells']}. The dataset is cited by DOI. Sex-only baseline, a majority-class baseline, a sex-stratified analysis with denominators and exact CIs, a test of whether acoustics predict sex (AUROC {V['sex_predictability_auroc']:.2f}) and a sex-adjusted sensitivity analysis are added (Sections 4.2, 4.5). Age is not in the public file, so a demographics-only baseline can only include sex; this is stated."),
 ("R1-5  SHAP reproducibility and alignment", "Our previous PermutationExplainer used max_evals = 2M+1, which corresponds to a single permutation per sample and gave unreliable rankings; we replaced it by exact TreeExplainer on the CV-selected XGBoost, explaining all 153 holdout recordings. Output (log-odds), background (none needed), unit (recording), direction, bootstrap CIs, fold-wise stability and local examples for TP/TN/FP/FN are reported (Sections 3.6, 4.6; Figs. 3–5). Text and figure now use the same ranking. The old KNN ranking is no longer reported. Attributions are described as model-specific; the ranking is only moderately stable and this is stated."),
 ("R1-6  Limit clinical claims", "Title changed as suggested; abstract, introduction, discussion and conclusion now describe an internal benchmark. Screening, early-detection and fairness language was removed or qualified. Diagnostic criteria, disease stage and medication are not in the public data and are listed as limitations. Participant flow (Fig. 1, Table 2), complete model settings (Table 3), software versions and a code link are provided. The related-work comparison now states which studies used this dataset and what is known about their protocol (Table 1). Ethics and data availability statements are in the manuscript body; a TRIPOD+AI checklist is to be uploaded."),
 ("R2-1  Subject-level validation and confusion matrices", "See R1-1; subject-level confusion matrices in Fig. 2 and Table 5."),
 ("R2-2  Data leakage", "Imputation and scaling are inside each scikit-learn pipeline and are fitted on training data only, in cross-validation and in the final fit (Section 3.3). We corrected the earlier Figure 1, which implied otherwise."),
 ("R2-3  Demographic confounding", "Addressed as far as the public data allow (Section 4.5): male controls are misclassified more often in all models, acoustics encode sex, and a sex-adjusted analysis does not remove the pattern. Age is not available in the public file and could not be analysed; this is a stated limitation."),
 ("R2-4  Interpretability", "Physiological interpretation of individual features was removed; we report direction, stability and association with sex, and state that SHAP does not establish biomarker validity (Sections 3.6, 4.6, 5)."),
 ("R2-5  Generalisability", "Single-dataset limitation and absence of external validation are stated throughout; claims about early detection and screening were removed."),
]
for t, a in resp:
    para(t, bold=True); para(a)
doc.save(OUT_R)
print("saved", OUT_M, OUT_R)
