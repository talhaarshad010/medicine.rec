"""
Medicine Recommendation System — Training & Preprocessing Script
================================================================
This script covers Modules 1–5:
  1. Data Loading & Validation
  2. Data Preprocessing
  3. Model Training & Evaluation
  4. Recommendation Engine
  5. Model Saving & Loading

Run:
    python medicine_recommendation.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_validate_data():
    """Load all four CSV files and print a validation summary.

    Returns
    -------
    df_main : pd.DataFrame
        Primary dataset (Disease + 17 symptom columns).
    df_severity : pd.DataFrame
        Symptom-to-severity-weight mapping.
    df_description : pd.DataFrame
        Disease descriptions.
    df_precaution : pd.DataFrame
        Disease precautions.
    """
    print("=" * 72)
    print("MODULE 1 — DATA LOADING & VALIDATION")
    print("=" * 72)

    # 1. Load CSVs
    df_main        = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    df_severity    = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
    df_description = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    df_precaution  = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))

    # 2. Strip all column names
    for df in [df_main, df_severity, df_description, df_precaution]:
        df.columns = df.columns.str.strip()

    # Strip string values in relevant columns
    df_main["Disease"] = df_main["Disease"].str.strip()
    symptom_cols = [c for c in df_main.columns if c.startswith("Symptom")]
    for col in symptom_cols:
        df_main[col] = df_main[col].astype(str).str.strip().replace("nan", np.nan)

    df_severity["Symptom"] = df_severity["Symptom"].str.strip()
    df_description["Disease"] = df_description["Disease"].str.strip()
    df_precaution["Disease"] = df_precaution["Disease"].str.strip()

    # 3. Print shapes
    print(f"\ndataset.csv          → {df_main.shape}")
    print(f"Symptom-severity.csv → {df_severity.shape}")
    print(f"symptom_Description  → {df_description.shape}")
    print(f"symptom_precaution   → {df_precaution.shape}")

    # 4. Unique diseases & symptoms in main dataset
    unique_diseases = df_main["Disease"].nunique()
    all_syms = set()
    for col in symptom_cols:
        all_syms.update(df_main[col].dropna().unique())
    print(f"\nUnique diseases in dataset.csv : {unique_diseases}")
    print(f"Unique symptoms in dataset.csv: {len(all_syms)}")

    # 5. Null counts per column
    print(f"\nNull counts per column (dataset.csv):")
    print(df_main.isnull().sum().to_string())

    # 6. Class balance
    class_counts = df_main["Disease"].value_counts()
    print(f"\nMin class count: {class_counts.min()}")
    print(f"Max class count: {class_counts.max()}")

    # 7. Weight range
    print(f"\nSeverity weight range: {df_severity['weight'].min()} – {df_severity['weight'].max()}")

    # 8. Cross-file disease coverage
    diseases_main = set(df_main["Disease"].unique())
    diseases_desc = set(df_description["Disease"].unique())
    diseases_prec = set(df_precaution["Disease"].unique())
    print(f"\nDiseases in description file : {len(diseases_desc)}")
    print(f"Diseases in precaution file : {len(diseases_prec)}")
    missing_desc = diseases_main - diseases_desc
    missing_prec = diseases_main - diseases_prec
    if missing_desc:
        print(f"  ⚠ Missing from description: {missing_desc}")
    if missing_prec:
        print(f"  ⚠ Missing from precaution : {missing_prec}")

    print("\n✅ Data loading & validation complete.\n")
    return df_main, df_severity, df_description, df_precaution


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_data(df_main, df_severity):
    """Clean and transform raw data into an ML-ready feature matrix.

    Parameters
    ----------
    df_main : pd.DataFrame
        Primary dataset.
    df_severity : pd.DataFrame
        Symptom-severity mapping.

    Returns
    -------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Encoded target vectors.
    label_encoder : LabelEncoder
        Fitted encoder for disease labels.
    all_symptoms : list[str]
        Sorted list of all unique symptom names.
    severity_dict : dict
        {symptom_name: weight} mapping.
    """
    print("=" * 72)
    print("MODULE 2 — DATA PREPROCESSING")
    print("=" * 72)

    # 1. Build severity dictionary
    severity_dict = dict(
        zip(df_severity["Symptom"].str.strip(), df_severity["weight"])
    )
    print(f"\nSeverity dictionary entries: {len(severity_dict)}")

    # 2. Collect all unique symptom names
    symptom_cols = [c for c in df_main.columns if c.startswith("Symptom")]
    all_symptoms_set = set()
    for col in symptom_cols:
        vals = df_main[col].dropna().str.strip()
        all_symptoms_set.update(vals)
    all_symptoms = sorted(all_symptoms_set)
    print(f"All unique symptoms (sorted) : {len(all_symptoms)}")

    # 3. Build feature matrix using severity weights
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
    n_samples = len(df_main)
    n_features = len(all_symptoms)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for row_idx in range(n_samples):
        for col in symptom_cols:
            val = df_main.iloc[row_idx][col]
            if pd.notna(val):
                sym = str(val).strip()
                if sym in symptom_to_idx:
                    weight = severity_dict.get(sym, 1)  # default 1 if missing
                    X[row_idx, symptom_to_idx[sym]] = weight

    # 4. Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_main["Disease"].str.strip())

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Number of classes    : {len(label_encoder.classes_)}")
    print(f"Train size           : {X_train.shape[0]}")
    print(f"Test size            : {X_test.shape[0]}")
    print("\n✅ Preprocessing complete.\n")

    return X_train, X_test, y_train, y_test, label_encoder, all_symptoms, severity_dict


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, all_symptoms):
    """Train a Random Forest classifier and evaluate it.

    Also trains Decision Tree and Multinomial Naive Bayes for comparison.

    Parameters
    ----------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    label_encoder : LabelEncoder
    all_symptoms : list[str]

    Returns
    -------
    rf : RandomForestClassifier
        Trained Random Forest model.
    """
    print("=" * 72)
    print("MODULE 3 — MODEL TRAINING & EVALUATION")
    print("=" * 72)

    # ── Primary model — Random Forest ──────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\n🌲 Random Forest Accuracy : {acc_rf * 100:.2f}%")

    # Full classification report
    class_names = label_encoder.classes_
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(18, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Random Forest", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n📊 Confusion matrix saved → {cm_path}")

    # Top 10 feature importances
    importances = rf.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    print("\n── Top 10 Most Important Symptoms ──")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:>2}. {all_symptoms[idx]:<30s} importance={importances[idx]:.4f}")

    # ── Secondary models ───────────────────────────────────────────────
    print("\n── Secondary Model Comparison ──")

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = accuracy_score(y_test, dt.predict(X_test))
    print(f"  Decision Tree Accuracy     : {acc_dt * 100:.2f}%")

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    acc_mnb = accuracy_score(y_test, mnb.predict(X_test))
    print(f"  Multinomial NB Accuracy    : {acc_mnb * 100:.2f}%")

    print(f"\n  ✅ Random Forest ({acc_rf*100:.2f}%) vs "
          f"Decision Tree ({acc_dt*100:.2f}%) vs "
          f"Multinomial NB ({acc_mnb*100:.2f}%)")
    print("\n✅ Model training & evaluation complete.\n")

    return rf


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_predict_function(rf, label_encoder, all_symptoms, severity_dict,
                           df_description, df_precaution):
    """Build and return a predict() closure that encapsulates all lookups.

    Parameters
    ----------
    rf : RandomForestClassifier
    label_encoder : LabelEncoder
    all_symptoms : list[str]
    severity_dict : dict
    df_description : pd.DataFrame
    df_precaution : pd.DataFrame

    Returns
    -------
    predict : callable
        predict(symptoms_input: list[str]) -> dict
    """
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
    importances = rf.feature_importances_
    top5_idx = np.argsort(importances)[::-1][:5]
    top5_symptoms = [all_symptoms[i] for i in top5_idx]

    # Pre-build lookup dicts for O(1) access
    desc_dict = dict(
        zip(df_description["Disease"].str.strip(),
            df_description["Description"].str.strip())
    )
    prec_dict = {}
    for _, row in df_precaution.iterrows():
        disease = str(row["Disease"]).strip()
        precs = []
        for i in range(1, 5):
            col = f"Precaution_{i}"
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                precs.append(str(row[col]).strip())
        prec_dict[disease] = precs

    def predict(symptoms_input):
        """Predict disease and return a structured recommendation.

        Parameters
        ----------
        symptoms_input : list[str]
            List of symptom name strings, e.g. ['itching', 'skin_rash'].

        Returns
        -------
        dict with keys:
            disease        – predicted disease name (str)
            description    – disease description paragraph (str)
            precautions    – list of 4 precaution strings (list)
            top_symptoms   – top 5 most important symptoms globally (list)
            severity_score – sum of severity weights of input symptoms (int)
        """
        # Build feature vector
        x = np.zeros(len(all_symptoms), dtype=np.float32)
        severity_score = 0
        for sym in symptoms_input:
            sym = sym.strip()
            if sym in symptom_to_idx:
                weight = severity_dict.get(sym, 1)
                x[symptom_to_idx[sym]] = weight
                severity_score += weight

        # Predict
        pred_idx = rf.predict(x.reshape(1, -1))[0]
        disease_name = label_encoder.inverse_transform([pred_idx])[0]

        # Look up description & precautions
        description = desc_dict.get(disease_name, "Description not available.")
        precautions = prec_dict.get(disease_name, ["Precautions not available."])

        return {
            "disease": disease_name,
            "description": description,
            "precautions": precautions,
            "top_symptoms": top5_symptoms,
            "severity_score": int(severity_score),
        }

    return predict


def test_recommendation_engine(predict_fn):
    """Run predict() on three different symptom combinations and print results.

    Parameters
    ----------
    predict_fn : callable
        The predict function returned by build_predict_function.
    """
    print("=" * 72)
    print("MODULE 4 — RECOMMENDATION ENGINE (Tests)")
    print("=" * 72)

    test_cases = [
        ["itching", "skin_rash", "nodal_skin_eruptions"],
        ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
        ["stomach_pain", "acidity", "vomiting"],
    ]

    for i, symptoms in enumerate(test_cases, 1):
        result = predict_fn(symptoms)
        print(f"\n── Test {i} ──")
        print(f"  Input symptoms  : {symptoms}")
        print(f"  Predicted disease: {result['disease']}")
        print(f"  Severity score   : {result['severity_score']} / 70")
        print(f"  Description      : {result['description'][:120]}...")
        print(f"  Precautions      : {result['precautions']}")
        print(f"  Top 5 symptoms   : {result['top_symptoms']}")

    print("\n✅ Recommendation engine tests complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MODEL SAVING & LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def save_model_artifacts(rf, label_encoder, all_symptoms, severity_dict):
    """Save trained model and supporting objects to the models/ directory.

    Parameters
    ----------
    rf : RandomForestClassifier
    label_encoder : LabelEncoder
    all_symptoms : list[str]
    severity_dict : dict
    """
    print("=" * 72)
    print("MODULE 5 — MODEL SAVING")
    print("=" * 72)

    artifacts = {
        "model.pkl": rf,
        "label_encoder.pkl": label_encoder,
        "symptoms_list.pkl": all_symptoms,
        "severity_dict.pkl": severity_dict,
    }

    for fname, obj in artifacts.items():
        path = os.path.join(MODEL_DIR, fname)
        joblib.dump(obj, path)
        print(f"  💾 Saved → {path}")

    print("\n✅ All model artifacts saved.\n")


def verify_saved_model(predict_fn):
    """Load saved artifacts and verify prediction matches the live model.

    Parameters
    ----------
    predict_fn : callable
        The predict function built from the live (in-memory) model.
    """
    print("── Verification: Load & Compare ──")

    rf_loaded           = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    le_loaded           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    all_symptoms_loaded = joblib.load(os.path.join(MODEL_DIR, "symptoms_list.pkl"))
    sev_dict_loaded     = joblib.load(os.path.join(MODEL_DIR, "severity_dict.pkl"))

    # Rebuild a quick prediction from loaded artefacts
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms_loaded)}
    test_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    x = np.zeros(len(all_symptoms_loaded), dtype=np.float32)
    for sym in test_symptoms:
        if sym in symptom_to_idx:
            x[symptom_to_idx[sym]] = sev_dict_loaded.get(sym, 1)

    pred_idx = rf_loaded.predict(x.reshape(1, -1))[0]
    disease = le_loaded.inverse_transform([pred_idx])[0]

    live_result = predict_fn(test_symptoms)

    print(f"  Loaded model prediction : {disease}")
    print(f"  Live model prediction   : {live_result['disease']}")
    match = disease == live_result["disease"]
    print(f"  Match: {'✅ YES' if match else '❌ NO'}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — run all modules in order
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Module 1
    df_main, df_severity, df_description, df_precaution = load_and_validate_data()

    # Module 2
    (X_train, X_test, y_train, y_test,
     label_encoder, all_symptoms, severity_dict) = preprocess_data(df_main, df_severity)

    # Module 3
    rf = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, all_symptoms)

    # Module 4
    predict_fn = build_predict_function(
        rf, label_encoder, all_symptoms, severity_dict, df_description, df_precaution
    )
    test_recommendation_engine(predict_fn)

    # Module 5
    save_model_artifacts(rf, label_encoder, all_symptoms, severity_dict)
    verify_saved_model(predict_fn)

    print("🎉 All modules complete. Run 'streamlit run app.py' to launch the web app.")
