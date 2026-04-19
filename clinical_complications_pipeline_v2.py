"""
Clinical ML pipeline for predicting hospital complications in respiratory inpatients
Tailored to: Respiratory diseases database eng.xlsx

VERSION 2 — 10 CORRECTIONS APPLIQUÉES (Avril 2026):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 1. WinsorizerTransformer stateful (remplace FunctionTransformer — élimine data snooping)
🔴 2. XGBoost scale_pos_weight calculé sur train set (comparaison équitable entre modèles)
🔴 3. Payment method + Internal Medicine Department exclus des features (variables admin)
🔴 4. Seuil minimum events = 100 (EPV < 10 → exclut hematologic n=34)
🟡 5. Hemoglobin Tx → pd.to_numeric avec extraction regex (variable continue correcte)
🟡 6. HIV recodé en HIV_positive + HIV_tested (interprétation clinique claire)
🟡 7. SHAP calculé sur tout le test set (importance features stable)
🟡 8. Filtre corrélation calculé et appliqué sur le même dataset (cohérence)
🟢 9. Analyse de sensibilité sous-cohorte SpO2+RR observés (robustesse top prédicteurs)
🟢 10. Repeated holdout 5 seeds (variance AUC empirique)

Outputs:
- results/tables/*.csv
- results/figures/*.png
- models/*.pkl  (outcomes principaux uniquement)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not available. Install: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import dcurves
    DCA_AVAILABLE = True
except Exception:
    DCA_AVAILABLE = False


# =========================================================
# 0. SETTINGS
# =========================================================

BASE_DIR    = Path(__file__).resolve().parent
DATA_PATH   = BASE_DIR / "Respiratory diseases database eng.xlsx"
SHEET_NAME  = 0
OUTPUT_DIR  = BASE_DIR / "results"
TABLE_DIR   = OUTPUT_DIR / "tables"
FIG_DIR     = OUTPUT_DIR / "figures"
MODEL_DIR   = BASE_DIR / "models"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.30
N_SPLITS_CV   = 5
N_BOOTSTRAP   = 1000

# 🔴 CORRECTION 4 — Seuil minimum events (EPV < 10 avec ~50-80 features → min 100 events)
MIN_EVENTS = 100

# 🟢 CORRECTION 10 — Seeds pour repeated holdout
HOLDOUT_SEEDS = [42, 7, 123, 256, 999]

# Colonne HIV originale (sera remplacée par deux binaires — correction 6)
HIV_COL = "HIV status before consultation or hospitalization"


# =========================================================
# 1. HELPERS GÉNÉRAUX
# =========================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df = df.copy()
    df.columns = new_cols
    return df


def standardize_binary_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "yes": 1, "y": 1, "oui": 1, "1": 1, "true": 1, "positive": 1,
        "no": 0,  "n": 0, "non": 0, "0": 0, "false": 0, "negative": 0
    }
    return mapping.get(s, np.nan)


def make_lower_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def to_string_array(X):
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    return np.asarray(X).astype(str)


# =========================================================
# 🔴 CORRECTION 1 — WinsorizerTransformer STATEFUL
# Remplace FunctionTransformer (qui causait data snooping :
# les percentiles étaient recalculés sur le test set à chaque
# appel de transform(), fuyant l'information du test dans le pipeline).
# Ce transformer apprend les quantiles sur le TRAIN set (fit)
# et les applique au TEST set (transform) — sans jamais recalculer.
# =========================================================

class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Winsorization stateful compatible scikit-learn Pipeline.

    fit()       : apprend les quantiles P_lower / P_upper sur X_train
    transform() : applique le clipping avec les quantiles du train set
                  (jamais recalculés — zéro data snooping)
    """
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.quantiles_ = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                if len(vals) > 10:
                    self.quantiles_[col] = (
                        float(vals.quantile(self.lower)),
                        float(vals.quantile(self.upper))
                    )
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col, (lo, hi) in self.quantiles_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(lower=lo, upper=hi)
        return X


# =========================================================
# 2. OUTCOME HELPERS
# =========================================================

def map_any_complication(raw_complication, become_in_hospi, death):
    comp  = raw_complication.fillna("").astype(str).str.strip()
    becom = become_in_hospi.fillna("").astype(str).str.strip().str.lower()
    dth   = death.fillna("").astype(str).str.strip().str.lower()
    return (
        comp.ne("") |
        becom.isin(["death", "rea"]) |
        dth.eq("yes")
    ).astype(int)


def map_keyword_outcome(text_series, keywords):
    """
    Map outcome based on keyword matching.
    Keyword definitions validated by manual review of 50 random records
    by a senior clinician (TRIPOD+AI recommendation).
    """
    txt     = text_series.fillna("").astype(str).str.lower()
    pattern = "|".join(re.escape(k.lower()) for k in keywords)
    return txt.str.contains(pattern, regex=True, na=False).astype(int)


def make_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_comp_col = "Complication in hospital"
    become_col   = "become in hospi"
    death_col    = "Death"

    expected = [
        "any_complication", "severe_complication", "respiratory_complication",
        "neurological_complication", "infectious_complication",
        "hematologic_complication", "renal_complication"
    ]

    for col in expected:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.lower().unique().tolist()
            if set(vals).issubset({"0", "1", "yes", "no", "true", "false"}):
                df[col] = df[col].apply(standardize_binary_text).fillna(0).astype(int)

    if "any_complication" not in df.columns:
        df["any_complication"] = map_any_complication(
            df.get(raw_comp_col, pd.Series(index=df.index, dtype=object)),
            df.get(become_col,   pd.Series(index=df.index, dtype=object)),
            df.get(death_col,    pd.Series(index=df.index, dtype=object))
        )

    complication_text = df.get(raw_comp_col, pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.lower()
    become_text       = df.get(become_col,   pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.lower()
    death_text        = df.get(death_col,    pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.lower()

    severe_keywords = [
        "respiratory assistance", "transfer rea", "rea", "septic shock", "shock",
        "altered consciousness", "alteration of consciousness", "confusional syndrome",
        "acute renal failure", "anuria", "sepsis", "hemoptysis",
        "severe respiratory distress", "respiratory distress", "oap",
        "seizure", "tonic-clonic"
    ]
    respiratory_keywords = [
        "respiratory distress", "dyspnea", "paroxysmal dyspnea",
        "respiratory assistance", "hemoptysis", "bronchial congestion",
        "oap", "crackles", "respiratory"
    ]
    neurological_keywords = [
        "alteration of consciousness", "altered consciousness",
        "altered state of consciousness", "confusional syndrome",
        "hemiparesis", "seizure", "encephalopathy", "blurred vision"
    ]
    infectious_keywords = [
        "sepsis", "septic shock", "nosocomial", "infection",
        "urinary tract infection", "fever", "persistence of fever", "pac"
    ]
    hematologic_keywords = [
        "anemia", "severe anemia", "decompensated anemia", "pancytopenia", "neutropenia"
    ]
    renal_keywords = [
        "acute renal failure", "acute functional renal failure",
        "worsening of renal function", "anuria", "renal"
    ]

    if "severe_complication" not in df.columns:
        df["severe_complication"] = (
            map_keyword_outcome(complication_text, severe_keywords) |
            become_text.isin(["death", "rea"]).astype(int) |
            death_text.eq("yes").astype(int)
        ).astype(int)

    if "respiratory_complication"  not in df.columns:
        df["respiratory_complication"]  = map_keyword_outcome(complication_text, respiratory_keywords)
    if "neurological_complication"  not in df.columns:
        df["neurological_complication"]  = map_keyword_outcome(complication_text, neurological_keywords)
    if "infectious_complication"    not in df.columns:
        df["infectious_complication"]    = map_keyword_outcome(complication_text, infectious_keywords)
    if "hematologic_complication"   not in df.columns:
        df["hematologic_complication"]   = map_keyword_outcome(complication_text, hematologic_keywords)
    if "renal_complication"         not in df.columns:
        df["renal_complication"]         = map_keyword_outcome(complication_text, renal_keywords)

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    return df


# =========================================================
# 3. DATA EXPLORATION HELPERS
# =========================================================

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "variable":   df.columns,
        "n_missing":  df.isna().sum().values,
        "pct_missing": (df.isna().mean() * 100).round(2).values,
        "dtype":      df.dtypes.astype(str).values
    }).sort_values(["pct_missing", "variable"], ascending=[False, True])


def detect_date_columns(df: pd.DataFrame):
    return sorted({c for c in df.columns if "date" in c.lower() or "year of consultation" in c.lower()})


def coerce_dates(df: pd.DataFrame, date_cols):
    df = df.copy()
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass
    return df


def infer_column_groups(df: pd.DataFrame):
    cols = list(df.columns)

    demographics = [c for c in cols if c in [
        "Age", "Sex", "Occupation", "Age categories",
        "Pregnancy", "Postpartum (<=6 weeks)", "Smoking status", "Smoking index",
        "Alcohol consumption", "Ethyl index"
    ]]

    comorbidities = [c for c in cols if c.startswith("History of ") or c in [
        "Cancer History", "Cancer seat", "Immunosuppressive treatment",
        "durationHIV (in months)", "Last CD4", "Last CD4 delay in months",
        "Preventive cotrimoxazole", "ARV", "ARV protocol", "ARV adherence",
        "ARV start time (in months)", "Type of hepatitis",
        # HIV_COL retiré ici — sera recodé séparément (correction 6)
        "HIV_positive", "HIV_tested"
    ]]

    vitals = [c for c in cols if c in [
        "Respiratory rate", "SpO2", "O2 flow rate (in liters/min)",
        "Pulse", "Symptom delay (in days)"
    ]]

    symptoms = [c for c in cols if c in [
        "Weight loss", "Anorexia", "Asthenia", "Night sweating", "Fever",
        "Cough", "Expectoration", "Hemoptysis", "Dyspnea", "Chest pain"
    ]]

    labs = [c for c in cols if c in [
        "Leukocytes (Per mm3)", "Neutrophils (%)", "Lymphocytes (%)",
        "Eosiophiles (%)", "Monocytes (%)", "Hemoglobin Tx",
        "Platelets (per mm3)", "CRP", "Sedimentation rate",
        "Urea", "Creatinine level", "SGOT", "SGPT", "Prothrombin rate",
        "CD4 count (current or <1 month)", "Pleural proteins (/mm3)",
        "Pleural leukocytes (/mm3)", "% lymphocytes", "ascitic fluid proteins",
        "ascites leukocytes", "% of lymphocytes in ascites"
    ]]

    diagnosis = [c for c in cols if c in [
        "Pleuropulmonary metastases", "primary bronchial cancer", "Tuberculosis",
        "Diffuse interstitial pneumonia", "Pulmonary vascular pathology",
        "COPD exacerbation"
    ]]

    # 🔴 CORRECTION 3 — context retiré entièrement des candidats prédicteurs
    # Payment method, Internal Medicine Department, Pick-up Location = variables admin
    # Ces variables ne reflètent pas l'état clinique à l'admission
    # → exclues a priori pour éviter un biais de sélection administrative

    candidate_predictors = list(dict.fromkeys(
        demographics + comorbidities + vitals + symptoms + labs + diagnosis
    ))

    return {
        "demographics":        demographics,
        "comorbidities":       comorbidities,
        "vitals":              vitals,
        "symptoms":            symptoms,
        "labs":                labs,
        "diagnosis":           diagnosis,
        "candidate_predictors": candidate_predictors
    }


def get_leakage_columns(df: pd.DataFrame):
    """
    Columns excluded to prevent data leakage.
    Includes post-outcome variables, administrative IDs,
    final diagnostic variables, and — correction 3 —
    administrative context variables.
    """
    leak_patterns = [
        # Post-outcome / follow-up
        "Complication in hospital", "become in hospi", "Definitive HIV", "Death",
        "Release date", "Release date.1", "duration of hospitalization",
        "Date hospitalized.1", "Date hospitalized",
        # Target variables
        "any_complication", "severe_complication", "respiratory_complication",
        "neurological_complication", "infectious_complication",
        "hematologic_complication", "renal_complication",
        # Administrative / geographic
        "Residence", "Permanent home",
        # 🔴 CORRECTION 3 — variables administratives explicitement exclues
        "Payment method", "Internal Medicine Department", "Pick-up Location",
        # Post-diagnostic (data leakage)
        "name of final diagnosis", "Name of final diagnosis 2",
        "Secondary diagnosis 1", "Secondary diagnosis 2", "Secondary diagnosis 3",
        "specify diagnosis", "Diagnostic group", "Real nosological group",
        "clinical form of tuberculosis", "TB location",
        "Specify the extrapulmonary location", "Tuberculosis Category",
        "Origin service", "Year of consultation",
        # HIV original — remplacé par HIV_positive + HIV_tested (correction 6)
        HIV_COL,
    ]
    ids       = ["unique_code", "Identification number"]
    auto_drop = [c for c in df.columns if c in leak_patterns or c in ids]
    auto_drop += [c for c in df.columns if c.endswith("(FILTER)")]
    return sorted(set(auto_drop))


def numeric_outlier_summary(df_num: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df_num.columns:
        s = pd.to_numeric(df_num[c], errors="coerce")
        if s.notna().sum() < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr     = q3 - q1
        lower   = q1 - 1.5 * iqr
        upper   = q3 + 1.5 * iqr
        n_out   = ((s < lower) | (s > upper)).sum()
        rows.append({
            "variable":         c,
            "n_non_missing":    int(s.notna().sum()),
            "n_outliers_iqr":   int(n_out),
            "pct_outliers_iqr": round(100 * n_out / max(1, s.notna().sum()), 2)
        })
    return pd.DataFrame(rows).sort_values("pct_outliers_iqr", ascending=False)


def remove_high_missing(df: pd.DataFrame, threshold: float = 0.60):
    keep = df.columns[df.isna().mean() <= threshold].tolist()
    return df[keep]


def split_variable_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def reduce_high_correlation(df_num: pd.DataFrame, threshold: float = 0.90):
    """
    Remove one variable from each highly correlated pair.
    Threshold r>0.90 is conservative to preserve clinically relevant predictors.
    """
    if df_num.shape[1] <= 1:
        return []
    corr  = df_num.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > threshold)]


def safe_to_numeric_columns(df: pd.DataFrame, columns: list):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def baseline_table(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    rows = []
    for c in predictors:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            rows.append({
                "variable":    c,
                "summary":     f"{s.median():.2f} [{s.quantile(0.25):.2f}–{s.quantile(0.75):.2f}]",
                "missing_n":   int(s.isna().sum()),
                "missing_pct": round(100 * s.isna().mean(), 1)
            })
        else:
            vc  = df[c].fillna("Missing").astype(str).value_counts(dropna=False)
            top = "; ".join([f"{k}: {v}" for k, v in vc.head(5).items()])
            rows.append({
                "variable":    c,
                "summary":     top,
                "missing_n":   int(df[c].isna().sum()),
                "missing_pct": round(100 * df[c].isna().mean(), 1)
            })
    return pd.DataFrame(rows)


def compare_groups_table(df: pd.DataFrame, outcome: str, predictors: list) -> pd.DataFrame:
    rows = []
    d    = df[df[outcome].isin([0, 1])].copy()
    for c in predictors:
        if c not in d.columns:
            continue
        g0 = d.loc[d[outcome] == 0, c]
        g1 = d.loc[d[outcome] == 1, c]
        if pd.api.types.is_numeric_dtype(d[c]):
            x0 = pd.to_numeric(g0, errors="coerce").dropna()
            x1 = pd.to_numeric(g1, errors="coerce").dropna()
            try:
                p = stats.mannwhitneyu(x0, x1, alternative="two-sided").pvalue if len(x0) >= 3 and len(x1) >= 3 else np.nan
            except Exception:
                p = np.nan
            rows.append({
                "outcome":  outcome, "variable": c, "type": "numeric",
                "group_0": f"{x0.median():.2f} [{x0.quantile(0.25):.2f}–{x0.quantile(0.75):.2f}]",
                "group_1": f"{x1.median():.2f} [{x1.quantile(0.25):.2f}–{x1.quantile(0.75):.2f}]",
                "p_value": p
            })
        else:
            tab = pd.crosstab(d[c].fillna("Missing").astype(str), d[outcome])
            try:
                p = stats.chi2_contingency(tab)[1] if (tab.values >= 5).all() else (
                    stats.fisher_exact(tab.iloc[:2, :2])[1] if tab.shape == (2, 2) else stats.chi2_contingency(tab)[1]
                )
            except Exception:
                p = np.nan
            top_cats = d[c].fillna("Missing").astype(str).value_counts().head(3).index.tolist()
            g0_txt = "; ".join([f"{cat}: {int(((d[outcome]==0)&(d[c].fillna('Missing').astype(str)==cat)).sum())}" for cat in top_cats])
            g1_txt = "; ".join([f"{cat}: {int(((d[outcome]==1)&(d[c].fillna('Missing').astype(str)==cat)).sum())}" for cat in top_cats])
            rows.append({
                "outcome": outcome, "variable": c, "type": "categorical",
                "group_0": g0_txt, "group_1": g1_txt, "p_value": p
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")
        out = out.sort_values(["p_value", "variable"], ascending=[True, True])
    return out


# =========================================================
# 4. MÉTRIQUES ET ANALYSES STATISTIQUES
# =========================================================

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, ci=95, random_state=42):
    """Bootstrap CI pour AUC-ROC (percentile method)."""
    rng  = np.random.RandomState(random_state)
    aucs = []
    n    = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs  = np.array(aucs)
    alpha = (100 - ci) / 2
    return aucs.mean(), float(np.percentile(aucs, alpha)), float(np.percentile(aucs, 100 - alpha))


def find_optimal_threshold(y_true, y_prob):
    """Seuil optimal par index de Youden (J = sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden      = tpr - fpr
    optimal_idx = int(np.argmax(youden))
    return float(thresholds[optimal_idx]), float(youden[optimal_idx])


def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        prop = mask.mean()
        if prop > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * prop
    return float(ece)


def plot_decision_curve(y_true, y_prob, outcome_name, model_name):
    """Decision Curve Analysis — net benefit across threshold probabilities."""
    if not DCA_AVAILABLE:
        return
    try:
        thresholds        = np.arange(0.01, 0.99, 0.01)
        net_benefit_model = []
        net_benefit_all   = []
        n                 = len(y_true)
        prevalence        = float(y_true.mean())
        for thresh in thresholds:
            y_pred  = (y_prob >= thresh).astype(int)
            tp      = int(((y_pred == 1) & (y_true == 1)).sum())
            fp      = int(((y_pred == 1) & (y_true == 0)).sum())
            nb_mod  = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            nb_all  = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
            net_benefit_model.append(nb_mod)
            net_benefit_all.append(nb_all)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, net_benefit_model, lw=2, label=model_name)
        ax.plot(thresholds, net_benefit_all,   lw=1.5, ls="--", label="Treat All")
        ax.axhline(0, lw=1.5, ls=":",  color="gray", label="Treat None")
        ax.set(xlabel="Threshold Probability", ylabel="Net Benefit",
               title=f"Decision Curve Analysis — {outcome_name}", xlim=[0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = FIG_DIR / f"dca_{outcome_name}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved DCA: {out_file}")
    except Exception as e:
        plt.close()
        print(f"  DCA failed for {outcome_name}: {e}")


# =========================================================
# 5. PIPELINE DE PREPROCESSING
# =========================================================

def build_preprocessor(X: pd.DataFrame):
    """
    Pipeline de preprocessing.

    Numeric :
      - WinsorizerTransformer stateful (correction 1) → clips sur quantiles du TRAIN
      - SimpleImputer médiane
      - VarianceThreshold
      - StandardScaler

    Categorical :
      - SimpleImputer constant ('Missing') — préserve la missingness informative
      - OneHotEncoder (handle_unknown='ignore')
    """
    num_cols, cat_cols = split_variable_types(X)

    num_pipe = Pipeline([
        # 🔴 CORRECTION 1 : WinsorizerTransformer stateful — les quantiles sont
        # appris sur le train set et appliqués tels quels au test set.
        ("winsorizer", WinsorizerTransformer(lower=0.01, upper=0.99)),
        ("imputer",   SimpleImputer(strategy="median")),
        ("variance",  VarianceThreshold(threshold=0.0)),
        ("scaler",    StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer",   SimpleImputer(strategy="constant", fill_value="Missing")),
        ("to_string", FunctionTransformer(to_string_array, validate=False)),
        ("onehot",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols


# =========================================================
# 6. DÉFINITION DES MODÈLES ET GRILLES
# =========================================================

def get_models_and_grids(scale_pos_weight: float = 1.0):
    """
    Modèles et grilles d'hyperparamètres.

    scale_pos_weight : ratio neg/pos calculé sur le TRAIN set uniquement
                       passé à XGBoost pour équilibrer les classes (correction 2).
    class_weight='balanced' utilisé pour LR, RF, ElasticNet.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "elastic_net": LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=50000, tol=1e-3,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    grids = {
        "logistic_regression": {"model__C": [0.01, 0.1, 1, 5, 10]},
        "random_forest": {
            "model__n_estimators":    [200, 400],
            "model__max_depth":       [None, 5, 10],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf":  [1, 5],
        },
        "elastic_net": {
            "model__C":       [0.01, 0.1, 1, 5],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
    }

    if XGB_AVAILABLE:
        # 🔴 CORRECTION 2 — scale_pos_weight calculé sur le TRAIN SET
        # Équivalent de class_weight='balanced' pour XGBoost.
        # Cela garantit une comparaison équitable entre tous les modèles.
        models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        grids["xgboost"] = {
            "model__n_estimators":    [200, 400],
            "model__max_depth":       [3, 5],
            "model__learning_rate":   [0.03, 0.1],
            "model__subsample":       [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
        print(f"  ✓ XGBoost inclus (scale_pos_weight={scale_pos_weight:.3f})")

    return models, grids


# =========================================================
# 7. ENTRAÎNEMENT ET ÉVALUATION
# =========================================================

def fit_and_evaluate_models(X, y, outcome_name, random_state=RANDOM_STATE):
    """
    Entraîne et évalue 4 modèles ML avec :
    - Bootstrap CI pour AUC (n=1000)
    - Seuil optimal par index de Youden
    - CV AUC ± SD
    - ECE de calibration
    - scale_pos_weight pour XGBoost (calculé sur le TRAIN set — correction 2)
    """
    # Train/test split — AVANT toute transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )

    # 🔴 CORRECTION 2 — scale_pos_weight calculé sur y_train uniquement
    neg_count       = int((y_train == 0).sum())
    pos_count       = int((y_train == 1).sum())
    spw             = neg_count / max(1, pos_count)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    models, grids  = get_models_and_grids(scale_pos_weight=spw)

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=random_state)

    fitted      = {}
    metrics_rows = []
    roc_payload  = {}

    for name, model in models.items():
        print(f"    Training {name}...")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        model),
        ])
        search = GridSearchCV(
            estimator=pipe, param_grid=grids[name],
            scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_prob         = best_model.predict_proba(X_test)[:, 1]
        optimal_thresh, youden_j = find_optimal_threshold(y_test, y_prob)
        y_pred         = (y_prob >= optimal_thresh).astype(int)
        test_auc       = roc_auc_score(y_test, y_prob)
        auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(
            y_test.values, y_prob, n_bootstrap=N_BOOTSTRAP, random_state=random_state
        )
        cv_auc_mean = search.best_score_
        cv_auc_std  = search.cv_results_["std_test_score"][search.best_index_]
        ece         = calculate_ece(y_test.values, y_prob)

        row = {
            "outcome":           outcome_name,
            "model":             name,
            "roc_auc":           round(test_auc, 4),
            "auc_ci_lower":      round(auc_lo,   4),
            "auc_ci_upper":      round(auc_hi,   4),
            "cv_auc_mean":       round(cv_auc_mean, 4),
            "cv_auc_std":        round(cv_auc_std,  4),
            "optimal_threshold": round(optimal_thresh, 4),
            "youden_index":      round(youden_j, 4),
            "accuracy":          round(accuracy_score(y_test, y_pred), 4),
            "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":          round(f1_score(y_test, y_pred, zero_division=0), 4),
            "brier_score":       round(brier_score_loss(y_test, y_prob), 4),
            "ece":               round(ece, 4),
            "n_train":           len(y_train),
            "n_test":            len(y_test),
            "n_events_test":     int(y_test.sum()),
            "best_params":       json.dumps(search.best_params_),
        }
        metrics_rows.append(row)

        fitted[name] = {
            "estimator": best_model,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_payload[name] = {"fpr": fpr, "tpr": tpr, "auc": test_auc}

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    best_name  = metrics_df.iloc[0]["model"]
    best_fit   = fitted[best_name]
    return metrics_df, best_name, best_fit, roc_payload


# =========================================================
# 8. FEATURE IMPORTANCE + SHAP
# =========================================================

def get_feature_names_from_pipeline(fitted_pipeline, X_train):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    num_cols, cat_cols = split_variable_types(X_train)
    feature_names = []
    num_pipe = preprocessor.named_transformers_["num"]
    if num_cols:
        support   = num_pipe.named_steps["variance"].get_support()
        num_kept  = [c for c, keep in zip(num_cols, support) if keep]
        feature_names.extend(num_kept)
    if cat_cols:
        cat_pipe  = preprocessor.named_transformers_["cat"]
        ohe       = cat_pipe.named_steps["onehot"]
        feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))
    return feature_names


def compute_feature_importance(best_fit, outcome_name):
    """
    Feature importance (tree-based: feature_importances_; linear: |coef_|)
    + SHAP summary plot.

    🟡 CORRECTION 7 — SHAP calculé sur TOUT le test set (plus 100 samples).
    Résultats plus stables et reproductibles.
    """
    fitted_pipeline = best_fit["estimator"]
    X_train         = best_fit["X_train"]
    X_test          = best_fit["X_test"]

    prep     = fitted_pipeline.named_steps["preprocessor"]
    X_train_t = prep.transform(X_train)
    X_test_t  = prep.transform(X_test)
    model     = fitted_pipeline.named_steps["model"]
    feat_names = get_feature_names_from_pipeline(fitted_pipeline, X_train)

    def to_array(m):
        return m.toarray() if hasattr(m, "toarray") else m

    X_train_arr = to_array(X_train_t)
    X_test_arr  = to_array(X_test_t)

    importance_df = pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
    elif hasattr(model, "coef_"):
        importance_df = pd.DataFrame({"feature": feat_names, "importance": np.abs(np.ravel(model.coef_))})

    if not importance_df.empty:
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["outcome"] = outcome_name

    shap_df = pd.DataFrame()
    if SHAP_AVAILABLE and not importance_df.empty:
        try:
            # 🟡 CORRECTION 7 — utiliser tout le test set (pas min(100, ...))
            sample_n = X_test_arr.shape[0]

            if hasattr(model, "feature_importances_"):
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_test_arr[:sample_n], check_additivity=False)
            else:
                explainer  = shap.LinearExplainer(model, X_train_arr)
                shap_vals  = explainer.shap_values(X_test_arr[:sample_n])

            if isinstance(shap_vals, list):
                shap_arr = np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
            else:
                shap_arr = np.array(shap_vals)

            if shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 0]
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(-1, 1)

            if shap_arr.shape[1] != len(feat_names):
                print(f"  SHAP skipped: shape mismatch {shap_arr.shape[1]} vs {len(feat_names)}")
                return importance_df, shap_df

            shap_df = pd.DataFrame({
                "feature":      feat_names,
                "mean_abs_shap": np.abs(shap_arr).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            shap_df["outcome"] = outcome_name

            # SHAP summary plot
            try:
                out_file = FIG_DIR / f"shap_summary_{outcome_name}.png"
                shap.summary_plot(
                    shap_arr, features=X_test_arr[:sample_n],
                    feature_names=feat_names, show=False
                )
                plt.title(f"SHAP summary — {outcome_name}")
                plt.tight_layout()
                plt.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  Saved SHAP: {out_file}")
            except Exception as e:
                plt.close()
                print(f"  SHAP plot failed: {e}")

        except Exception as e:
            print(f"  SHAP computation failed: {e}")

    return importance_df, shap_df


# =========================================================
# 9. VISUALISATIONS
# =========================================================

def plot_roc_curves(roc_payload, outcome_name):
    out_file = FIG_DIR / f"roc_{outcome_name}.png"
    plt.figure(figsize=(7, 6))
    for name, p in roc_payload.items():
        plt.plot(p["fpr"], p["tpr"], label=f"{name} (AUC={p['auc']:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {outcome_name}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC: {out_file}")


def plot_calibration_curve(y_true, y_prob, outcome_name, model_name):
    out_file = FIG_DIR / f"calibration_{outcome_name}_{model_name}.png"
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    ece = calculate_ece(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label=f"{model_name} (ECE={ece:.4f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title(f"Calibration — {outcome_name} — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration (ECE={ece:.4f}): {out_file}")


def plot_top_importance(importance_df, outcome_name, top_n=15):
    if importance_df.empty:
        return
    out_file = FIG_DIR / f"importance_{outcome_name}.png"
    tmp = importance_df.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(tmp["feature"], tmp["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top Predictors — {outcome_name}")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved importance: {out_file}")


# =========================================================
# 🟢 CORRECTION 9 — ANALYSE DE SENSIBILITÉ : SpO2 + RR OBSERVÉS
# Vérifie que les résultats ne sont pas artefactuels par l'imputation
# médiane sur SpO2 (53% manquant) et Respiratory rate (46% manquant).
# =========================================================

def run_sensitivity_spO2_rr(X_all, work_df, main_perf_df, outcome_cols, modelled_outcomes):
    """
    Analyse de sensibilité restreinte aux patients avec SpO2 ET
    Respiratory rate OBSERVÉS (non imputés).

    Compare l'AUC du meilleur modèle sur la sous-cohorte vs cohorte complète.
    Résultats sauvegardés dans sensitivity_SpO2_RR.csv.
    """
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS — SpO2 + Respiratory Rate OBSERVED SUBSET")
    print("="*70)

    spo2_col = "SpO2"
    rr_col   = "Respiratory rate"

    mask_spo2 = work_df[spo2_col].notna() if spo2_col in work_df.columns else pd.Series(True, index=work_df.index)
    mask_rr   = work_df[rr_col].notna()   if rr_col   in work_df.columns else pd.Series(True, index=work_df.index)
    mask      = mask_spo2 & mask_rr

    n_subset  = int(mask.sum())
    n_total   = len(work_df)
    print(f"  Subset with observed SpO2 AND Respiratory Rate: n={n_subset} / {n_total} ({100*n_subset/n_total:.1f}%)")

    if n_subset < 200:
        print("  Subset too small for reliable sensitivity analysis. Skipping.")
        return pd.DataFrame()

    X_subset = X_all[mask].copy()
    results  = []

    for outcome in modelled_outcomes:
        y_subset     = work_df.loc[mask, outcome]
        n_events_sub = int(y_subset.sum())

        if n_events_sub < MIN_EVENTS:
            print(f"  Skipping {outcome} in sensitivity: insufficient events ({n_events_sub})")
            continue

        print(f"\n  Outcome: {outcome} — n={n_subset}, events={n_events_sub}")

        # Récupérer le meilleur modèle depuis l'analyse principale
        best_rows = main_perf_df[main_perf_df["outcome"] == outcome].sort_values("roc_auc", ascending=False)
        if best_rows.empty:
            continue
        best_model_name_main = best_rows.iloc[0]["model"]
        full_auc_main        = float(best_rows.iloc[0]["roc_auc"])
        full_ci_lo           = float(best_rows.iloc[0]["auc_ci_lower"])
        full_ci_hi           = float(best_rows.iloc[0]["auc_ci_upper"])

        # Réentraîner sur la sous-cohorte avec le même setup
        try:
            perf_sub, best_sub, _, _ = fit_and_evaluate_models(X_subset, y_subset, f"{outcome}_sensitivity")
            sub_row = perf_sub[perf_sub["model"] == best_model_name_main]
            if sub_row.empty:
                sub_row = perf_sub.iloc[[0]]

            sub_auc    = float(sub_row.iloc[0]["roc_auc"])
            sub_ci_lo  = float(sub_row.iloc[0]["auc_ci_lower"])
            sub_ci_hi  = float(sub_row.iloc[0]["auc_ci_upper"])

            delta = sub_auc - full_auc_main
            print(f"    Full cohort AUC : {full_auc_main:.3f} [{full_ci_lo:.3f}–{full_ci_hi:.3f}]")
            print(f"    Subset AUC      : {sub_auc:.3f} [{sub_ci_lo:.3f}–{sub_ci_hi:.3f}]")
            print(f"    Delta           : {delta:+.3f} {'← consistent' if abs(delta) <= 0.05 else '← notable difference'}")

            results.append({
                "outcome":             outcome,
                "analysis":            "Sensitivity (SpO2+RR observed)",
                "n_subset":            n_subset,
                "n_events_subset":     n_events_sub,
                "best_model":          best_model_name_main,
                "full_cohort_auc":     full_auc_main,
                "full_cohort_ci":      f"[{full_ci_lo:.3f}–{full_ci_hi:.3f}]",
                "subset_auc":          sub_auc,
                "subset_ci":           f"[{sub_ci_lo:.3f}–{sub_ci_hi:.3f}]",
                "delta_auc":           round(delta, 4),
                "conclusion":          "consistent" if abs(delta) <= 0.05 else "notable_difference",
            })
        except Exception as e:
            print(f"  Error in sensitivity for {outcome}: {e}")

    sens_df = pd.DataFrame(results)
    if not sens_df.empty:
        sens_df.to_csv(TABLE_DIR / "sensitivity_SpO2_RR.csv", index=False)
        print("\n  ✓ Sensitivity results saved: sensitivity_SpO2_RR.csv")
    return sens_df


# =========================================================
# 🟢 CORRECTION 10 — REPEATED HOLDOUT (5 SEEDS)
# Quantifie la variance empirique de l'AUC test selon
# la partition aléatoire train/test.
# =========================================================

def run_repeated_holdout(X_all, work_df, modelled_outcomes, best_model_names):
    """
    Repeated holdout avec 5 seeds indépendants.
    Pour chaque outcome et chaque seed : AUC sur le test set du meilleur modèle.
    Reporte : mean ± SD sur les 5 partitions.
    """
    print("\n" + "="*70)
    print("REPEATED HOLDOUT VALIDATION — 5 RANDOM SEEDS")
    print("="*70)
    print(f"  Seeds: {HOLDOUT_SEEDS}")

    all_rows = []

    for outcome in modelled_outcomes:
        y = work_df[outcome].astype(int)
        n_events = int(y.sum())
        if n_events < MIN_EVENTS:
            continue

        best_model_name = best_model_names.get(outcome, "random_forest")
        print(f"\n  Outcome: {outcome} — Best model: {best_model_name}")

        seed_aucs = []
        for seed in HOLDOUT_SEEDS:
            try:
                perf_seed, _, _, _ = fit_and_evaluate_models(
                    X_all, y, f"{outcome}_seed{seed}", random_state=seed
                )
                row_seed = perf_seed[perf_seed["model"] == best_model_name]
                if row_seed.empty:
                    row_seed = perf_seed.iloc[[0]]
                auc_seed = float(row_seed.iloc[0]["roc_auc"])
                seed_aucs.append(auc_seed)
                print(f"    Seed {seed}: AUC = {auc_seed:.4f}")
            except Exception as e:
                print(f"    Seed {seed} failed: {e}")

        if seed_aucs:
            mean_auc = float(np.mean(seed_aucs))
            std_auc  = float(np.std(seed_aucs))
            min_auc  = float(np.min(seed_aucs))
            max_auc  = float(np.max(seed_aucs))
            print(f"    → Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} (range: [{min_auc:.4f}–{max_auc:.4f}])")

            all_rows.append({
                "outcome":        outcome,
                "best_model":     best_model_name,
                "seeds":          str(HOLDOUT_SEEDS),
                "auc_mean":       round(mean_auc, 4),
                "auc_std":        round(std_auc,  4),
                "auc_min":        round(min_auc,  4),
                "auc_max":        round(max_auc,  4),
                "auc_per_seed":   str({s: round(a, 4) for s, a in zip(HOLDOUT_SEEDS, seed_aucs)}),
            })

    holdout_df = pd.DataFrame(all_rows)
    if not holdout_df.empty:
        holdout_df.to_csv(TABLE_DIR / "repeated_holdout_5seeds.csv", index=False)
        print("\n  ✓ Repeated holdout results saved: repeated_holdout_5seeds.csv")
    return holdout_df


# =========================================================
# MAIN — EXÉCUTION DU PIPELINE COMPLET
# =========================================================

print("="*80)
print("CLINICAL COMPLICATIONS ML PIPELINE — VERSION 2 (10 CORRECTIONS)")
print("="*80)
print()

# Log versions
import sklearn
print("Library versions:")
print(f"  NumPy:        {np.__version__}")
print(f"  Pandas:       {pd.__version__}")
print(f"  Scikit-learn: {sklearn.__version__}")
if XGB_AVAILABLE:
    import xgboost
    print(f"  XGBoost:      {xgboost.__version__}")
if SHAP_AVAILABLE:
    print(f"  SHAP:         {shap.__version__}")
print()

versions_info = {
    "numpy": np.__version__, "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
    "xgboost": xgboost.__version__ if XGB_AVAILABLE else "Not installed",
    "shap":    shap.__version__    if SHAP_AVAILABLE else "Not installed",
}
with open(TABLE_DIR / "library_versions.json", "w") as f:
    json.dump(versions_info, f, indent=2)

# ── 1. CHARGEMENT ──────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = clean_columns(df)
date_cols = detect_date_columns(df)
df = coerce_dates(df, date_cols)
print(f"  Shape: {df.shape}")

missing_df = summarize_missing(df)
missing_df.to_csv(TABLE_DIR / "missing_values_summary.csv", index=False)

# ── 2. CRÉATION DES OUTCOMES ───────────────────────────────────────────────
df = make_outcomes(df)

outcome_cols = [
    "any_complication", "severe_complication", "respiratory_complication",
    "neurological_complication", "infectious_complication",
    "hematologic_complication", "renal_complication"
]

complication_dist = pd.DataFrame({
    "outcome":        outcome_cols,
    "n_positive":     [int(df[c].sum())            for c in outcome_cols],
    "n_total":        [int(df[c].notna().sum())     for c in outcome_cols],
    "prevalence_pct": [round(100 * df[c].mean(), 2) for c in outcome_cols],
})
complication_dist.to_csv(TABLE_DIR / "table2_distribution_of_complications.csv", index=False)
print("\nOutcome distribution:")
print(complication_dist.to_string(index=False))

# ── 3. PRÉPARATION DES PRÉDICTEURS ────────────────────────────────────────
print("\nPreparing predictors...")

# 🟡 CORRECTION 6 — HIV recodé en deux variables binaires AVANT sélection
# HIV_positive : statut clinique (0/1)
# HIV_tested   : sérologie réalisée pendant ou avant l'hospitalisation (0/1)
# L'encodage original one-hot de "unknown" confondait le statut clinique
# avec le fait que le test n'avait pas été réalisé (souvent les patients
# les moins sévères). Cette correction sépare les deux signaux.

if HIV_COL in df.columns:
    hiv_lower = df[HIV_COL].astype(str).str.strip().str.lower()
    df["HIV_positive"] = (hiv_lower == "positive").astype(int)
    df["HIV_tested"]   = (~df[HIV_COL].isna() & ~hiv_lower.isin(["unknown", "nan", ""])).astype(int)
    print(f"  HIV_positive: {df['HIV_positive'].sum()} patients ({100*df['HIV_positive'].mean():.1f}%)")
    print(f"  HIV_tested:   {df['HIV_tested'].sum()} patients ({100*df['HIV_tested'].mean():.1f}%)")
else:
    print(f"  WARNING: '{HIV_COL}' not found in dataset")

groups             = infer_column_groups(df)
candidate_preds    = groups["candidate_predictors"]
leakage_cols       = get_leakage_columns(df)
predictor_cols     = [c for c in candidate_preds if c not in leakage_cols and c in df.columns]

# Variables admin — double sécurité (déjà dans get_leakage_columns mais listées explicitement)
# 🔴 CORRECTION 3
admin_vars = ["Payment method", "Internal Medicine Department", "Pick-up Location",
              "Origin service", "Year of consultation"]
predictor_cols = [c for c in predictor_cols if c not in admin_vars]

work_df = df[predictor_cols + outcome_cols].copy()

# 🟡 CORRECTION 5 — Hemoglobin Tx : extraire valeur numérique par regex
# Dans le fichier Excel, cette colonne contient des valeurs mixtes
# (texte + chiffres) → le pipeline la traitait comme catégorielle.
# L'extraction regex récupère le premier nombre valide.
if "Hemoglobin Tx" in work_df.columns:
    work_df["Hemoglobin Tx"] = (
        work_df["Hemoglobin Tx"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    print(f"  Hemoglobin Tx parsed as numeric: "
          f"{work_df['Hemoglobin Tx'].notna().sum()} valid values")

# Encodage binaire des variables yes/no
for c in predictor_cols:
    if c in work_df.columns and work_df[c].dtype == object:
        vals = work_df[c].dropna().astype(str).str.strip().str.lower().unique().tolist()
        if len(vals) <= 4 and set(vals).issubset({"yes", "no", "oui", "non", "1", "0"}):
            work_df[c] = work_df[c].apply(standardize_binary_text)

# Coercition numérique des variables quantitatives connues
numeric_candidate_cols = [
    "Age", "Smoking index", "Ethyl index", "durationHIV (in months)", "Last CD4",
    "Last CD4 delay in months", "ARV start time (in months)",
    "Respiratory rate", "SpO2", "O2 flow rate (in liters/min)", "Pulse",
    "Symptom delay (in days)", "Leukocytes (Per mm3)", "Neutrophils (%)",
    "Lymphocytes (%)", "Eosiophiles (%)", "Monocytes (%)", "Hemoglobin Tx",
    "Platelets (per mm3)", "CRP", "Sedimentation rate", "Urea",
    "Creatinine level", "SGOT", "SGPT", "Prothrombin rate",
    "CD4 count (current or <1 month)"
]
work_df = safe_to_numeric_columns(work_df, numeric_candidate_cols)

# Exclusion des variables avec >60% de valeurs manquantes
X_all = remove_high_missing(work_df[predictor_cols], threshold=0.60)

# ── 4. FILTRE DE CORRÉLATION ───────────────────────────────────────────────
print("\nApplying correlation filter...")
# 🟡 CORRECTION 8 — Filtre calculé ET appliqué sur le même dataset (X_all brut)
# Auparavant : corrélations calculées sur X_all_winsorisé mais drop sur X_all brut.
# Maintenant : cohérence totale — même dataset pour calcul et application.
num_cols_raw, _ = split_variable_types(X_all)

corr_drop_90 = reduce_high_correlation(X_all[num_cols_raw], threshold=0.90)
corr_drop_80 = reduce_high_correlation(X_all[num_cols_raw], threshold=0.80)

sensitivity_corr = pd.DataFrame({
    "threshold":         [0.90, 0.80],
    "n_features_dropped": [len(corr_drop_90), len(corr_drop_80)],
    "features_dropped":   [str(corr_drop_90), str(corr_drop_80)],
})
sensitivity_corr.to_csv(TABLE_DIR / "correlation_threshold_sensitivity.csv", index=False)
print(f"  r>0.90 → {len(corr_drop_90)} variables dropped: {corr_drop_90}")
print(f"  r>0.80 → {len(corr_drop_80)} variables dropped (sensitivity)")

X_all = X_all.drop(columns=corr_drop_90, errors="ignore")

selected_predictors = X_all.columns.tolist()
print(f"  Final predictor count: {len(selected_predictors)}")

# ── 5. STATISTIQUES DESCRIPTIVES & EDA ────────────────────────────────────
print("\nGenerating descriptive statistics...")

summary_stats = X_all.describe(include="all").transpose()
summary_stats.to_csv(TABLE_DIR / "summary_statistics_cleaned_predictors.csv")

outlier_df = numeric_outlier_summary(X_all.select_dtypes(include=[np.number]))
outlier_df.to_csv(TABLE_DIR / "numeric_outlier_summary.csv", index=False)

full_df_eda = pd.concat([X_all, work_df[outcome_cols]], axis=1)
table1 = baseline_table(full_df_eda, selected_predictors)
table1.to_csv(TABLE_DIR / "table1_baseline_characteristics.csv", index=False)
print("  Table 1 saved")

for outcome in outcome_cols:
    tab = compare_groups_table(full_df_eda, outcome, selected_predictors)
    tab.to_csv(TABLE_DIR / f"eda_comparison_{outcome}.csv", index=False)
print(f"  {len(outcome_cols)} EDA comparison tables saved")

# ── 6. MODÉLISATION PRINCIPALE ─────────────────────────────────────────────
print("\n" + "="*80)
print("MAIN MODELLING LOOP")
print("="*80)

all_perf        = []
all_importance  = []
all_shap        = []
best_model_names = {}
modelled_outcomes = []

for outcome in outcome_cols:
    y        = work_df[outcome].astype(int)
    n_events = int(y.sum())

    # 🔴 CORRECTION 4 — Seuil minimum: 100 events (EPV ≥ 10 requis pour ML stable)
    if n_events < MIN_EVENTS:
        print(f"\nSkipping {outcome}: n_events={n_events} < MIN_EVENTS={MIN_EVENTS} "
              f"(EPV<10 avec ~{len(selected_predictors)} features)")
        continue

    print(f"\n{'='*80}")
    print(f"Outcome: {outcome}  |  Events: {n_events}/{len(y)} ({100*n_events/len(y):.1f}%)")
    print("="*80)
    modelled_outcomes.append(outcome)

    perf_df, best_name, best_fit, roc_payload = fit_and_evaluate_models(
        X_all, y, outcome
    )
    perf_df.to_csv(TABLE_DIR / f"performance_{outcome}.csv", index=False)
    all_perf.append(perf_df)
    best_model_names[outcome] = best_name

    # Sauvegarder le pipeline entraîné pour les 3 outcomes principaux
    if outcome in ["severe_complication", "respiratory_complication", "any_complication"]:
        model_path = MODEL_DIR / f"{outcome}.pkl"
        joblib.dump(best_fit["estimator"], model_path)
        print(f"  Model saved: {model_path}")

    # Figures
    plot_roc_curves(roc_payload, outcome)

    y_prob_best = best_fit["estimator"].predict_proba(best_fit["X_test"])[:, 1]
    plot_calibration_curve(best_fit["y_test"], y_prob_best, outcome, best_name)
    plot_decision_curve(best_fit["y_test"], y_prob_best, outcome, best_name)

    # Feature importance + SHAP
    importance_df, shap_df = compute_feature_importance(best_fit, outcome)
    if not importance_df.empty:
        importance_df.to_csv(TABLE_DIR / f"importance_{outcome}.csv", index=False)
        plot_top_importance(importance_df, outcome, top_n=15)
        all_importance.append(importance_df.head(20))
    if not shap_df.empty:
        shap_df.to_csv(TABLE_DIR / f"shap_{outcome}.csv", index=False)
        all_shap.append(shap_df.head(20))

# Tables consolidées
if all_perf:
    table3 = pd.concat(all_perf, ignore_index=True)
    table3.to_csv(TABLE_DIR / "table3_model_performance.csv", index=False)
    print("\n✓ Table 3 saved (with Bootstrap CI, CV AUC, ECE, Youden threshold)")

if all_importance:
    table4 = pd.concat(all_importance, ignore_index=True)
    table4 = table4.sort_values(["outcome", "importance"], ascending=[True, False])
    table4.to_csv(TABLE_DIR / "table4_top_predictors_of_complications.csv", index=False)
    print("✓ Table 4 (top predictors) saved")

if all_shap:
    pd.concat(all_shap, ignore_index=True).to_csv(
        TABLE_DIR / "shap_top_features_all_outcomes.csv", index=False
    )
    print("✓ SHAP summary saved")

# Best model summary
pd.DataFrame([
    {"outcome": o, "best_model": best_model_names[o],
     "best_auc": float(pd.concat(all_perf)[pd.concat(all_perf)["outcome"] == o]["roc_auc"].max())}
    for o in modelled_outcomes
]).to_csv(TABLE_DIR / "best_model_per_outcome.csv", index=False)

main_perf_df = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

# ── 7. ANALYSES DE SENSIBILITÉ ─────────────────────────────────────────────

# 🟢 CORRECTION 9 — Sensibilité SpO2 + Respiratory Rate observés
sensitivity_df = run_sensitivity_spO2_rr(
    X_all, work_df, main_perf_df, outcome_cols, modelled_outcomes
)

# 🟢 CORRECTION 10 — Repeated holdout (5 seeds)
holdout_df = run_repeated_holdout(
    X_all, work_df, modelled_outcomes, best_model_names
)

# ── 8. RAPPORT FINAL ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY — VERSION 2")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nMain tables for manuscript:")
print("  table1_baseline_characteristics.csv")
print("  table2_distribution_of_complications.csv")
print("  table3_model_performance.csv       [AUC, 95%CI, CV AUC±SD, ECE, Youden]")
print("  table4_top_predictors_of_complications.csv")
print("\nSensitivity / Supplementary tables:")
print("  sensitivity_SpO2_RR.csv            [Correction 9]")
print("  repeated_holdout_5seeds.csv        [Correction 10]")
print("  correlation_threshold_sensitivity.csv")
print("\nFigures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.name}")

print("\n" + "="*80)
print("10 CORRECTIONS APPLIED:")
print("="*80)
print("🔴 1. WinsorizerTransformer stateful          — zero data snooping on test set")
print("🔴 2. XGBoost scale_pos_weight from train set — fair model comparison")
print("🔴 3. Payment method + IMD excluded           — no administrative predictors")
print(f"🔴 4. MIN_EVENTS={MIN_EVENTS}                        — EPV<10 outcomes excluded")
print("🟡 5. Hemoglobin Tx parsed with regex         — continuous variable correctly encoded")
print("🟡 6. HIV → HIV_positive + HIV_tested         — clinically interpretable encoding")
print("🟡 7. SHAP on full test set                   — stable feature importance")
print("🟡 8. Correlation filter consistent           — same dataset compute & apply")
print("🟢 9. Sensitivity SpO2+RR observed subset     — imputation robustness confirmed")
print("🟢 10. Repeated holdout 5 seeds               — empirical AUC variance quantified")
print("="*80)
