"""
Medicine Recommendation System — Test Suite
=============================================
Runs 25 test cases covering all major diseases with various symptom
combinations to verify the prediction engine.

Run:
    python test_cases.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")


def load_artifacts():
    """Load all persisted model artifacts and CSV lookups.

    Returns
    -------
    rf, le, all_symptoms, sev_dict, desc_dict, prec_dict
    """
    rf           = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    all_symptoms = joblib.load(os.path.join(MODEL_DIR, "symptoms_list.pkl"))
    sev_dict     = joblib.load(os.path.join(MODEL_DIR, "severity_dict.pkl"))

    df_desc = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    df_prec = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
    df_desc.columns = df_desc.columns.str.strip()
    df_prec.columns = df_prec.columns.str.strip()
    df_desc["Disease"] = df_desc["Disease"].str.strip()
    df_prec["Disease"] = df_prec["Disease"].str.strip()

    desc_dict = dict(zip(df_desc["Disease"], df_desc["Description"].str.strip()))
    prec_dict = {}
    for _, row in df_prec.iterrows():
        d = str(row["Disease"]).strip()
        precs = []
        for i in range(1, 5):
            v = row.get(f"Precaution_{i}")
            if pd.notna(v) and str(v).strip():
                precs.append(str(v).strip())
        prec_dict[d] = precs

    return rf, le, all_symptoms, sev_dict, desc_dict, prec_dict


def predict(symptoms_input, rf, le, all_symptoms, sev_dict, desc_dict, prec_dict):
    """Predict disease from a list of symptom strings.

    Parameters
    ----------
    symptoms_input : list[str]
    rf, le, all_symptoms, sev_dict, desc_dict, prec_dict : model artifacts

    Returns
    -------
    dict with disease, description, precautions, severity_score
    """
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
    x = np.zeros(len(all_symptoms), dtype=np.float32)
    severity_score = 0
    for sym in symptoms_input:
        sym = sym.strip()
        if sym in symptom_to_idx:
            weight = sev_dict.get(sym, 1)
            x[symptom_to_idx[sym]] = weight
            severity_score += weight

    pred_idx = rf.predict(x.reshape(1, -1))[0]
    disease_name = le.inverse_transform([pred_idx])[0]
    description = desc_dict.get(disease_name, "N/A")
    precautions = prec_dict.get(disease_name, ["N/A"])

    return {
        "disease": disease_name,
        "description": description,
        "precautions": precautions,
        "severity_score": int(severity_score),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 25 TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── 1. Fungal infection (classic)
    {
        "id": 1,
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
        "expected": "Fungal infection",
    },
    # ── 2. Allergy
    {
        "id": 2,
        "symptoms": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
        "expected": "Allergy",
    },
    # ── 3. GERD
    {
        "id": 3,
        "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
        "expected": "GERD",
    },
    # ── 4. Diabetes
    {
        "id": 4,
        "symptoms": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "excessive_hunger", "polyuria"],
        "expected": "Diabetes",
    },
    # ── 5. Malaria
    {
        "id": 5,
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        "expected": "Malaria",
    },
    # ── 6. Common Cold
    {
        "id": 6,
        "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "runny_nose", "congestion", "sinus_pressure", "loss_of_smell", "throat_irritation", "phlegm"],
        "expected": "Common Cold",
    },
    # ── 7. Dengue
    {
        "id": 7,
        "symptoms": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "pain_behind_the_eyes", "muscle_pain", "red_spots_over_body"],
        "expected": "Dengue",
    },
    # ── 8. Heart attack
    {
        "id": 8,
        "symptoms": ["vomiting", "breathlessness", "sweating", "chest_pain"],
        "expected": "Heart attack",
    },
    # ── 9. Pneumonia
    {
        "id": 9,
        "symptoms": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "chest_pain", "fast_heart_rate", "rusty_sputum"],
        "expected": "Pneumonia",
    },
    # ── 10. Typhoid
    {
        "id": 10,
        "symptoms": ["chills", "vomiting", "fatigue", "high_fever", "nausea", "constipation", "abdominal_pain", "diarrhoea", "belly_pain"],
        "expected": "Typhoid",
    },
    # ── 11. Hepatitis B
    {
        "id": 11,
        "symptoms": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "yellowing_of_eyes", "receiving_blood_transfusion"],
        "expected": "Hepatitis B",
    },
    # ── 12. Jaundice
    {
        "id": 12,
        "symptoms": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
        "expected": "Jaundice",
    },
    # ── 13. Urinary tract infection
    {
        "id": 13,
        "symptoms": ["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"],
        "expected": "Urinary tract infection",
    },
    # ── 14. Acne
    {
        "id": 14,
        "symptoms": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        "expected": "Acne",
    },
    # ── 15. Psoriasis
    {
        "id": 15,
        "symptoms": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
        "expected": "Psoriasis",
    },
    # ── 16. Arthritis
    {
        "id": 16,
        "symptoms": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
        "expected": "Arthritis",
    },
    # ── 17. Bronchial Asthma
    {
        "id": 17,
        "symptoms": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
        "expected": "Bronchial Asthma",
    },
    # ── 18. Migraine
    {
        "id": 18,
        "symptoms": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "visual_disturbances"],
        "expected": "Migraine",
    },
    # ── 19. Chicken pox
    {
        "id": 19,
        "symptoms": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "red_spots_over_body", "malaise"],
        "expected": "Chicken pox",
    },
    # ── 20. Dimorphic hemmorhoids(piles)
    {
        "id": 20,
        "symptoms": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        "expected": "Dimorphic hemmorhoids(piles)",
    },
    # ── 21. Hyperthyroidism
    {
        "id": 21,
        "symptoms": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "irritability", "abnormal_menstruation"],
        "expected": "Hyperthyroidism",
    },
    # ── 22. Gastroenteritis
    {
        "id": 22,
        "symptoms": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
        "expected": "Gastroenteritis",
    },
    # ── 23. Tuberculosis
    {
        "id": 23,
        "symptoms": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "blood_in_sputum", "phlegm"],
        "expected": "Tuberculosis",
    },
    # ── 24. Varicose veins
    {
        "id": 24,
        "symptoms": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
        "expected": "Varicose veins",
    },
    # ── 25. Paralysis (brain hemorrhage) — minimal symptoms
    {
        "id": 25,
        "symptoms": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
        "expected": "Paralysis (brain hemorrhage)",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Execute all 25 test cases and print a results table."""
    rf, le, all_symptoms, sev_dict, desc_dict, prec_dict = load_artifacts()

    total = len(TEST_CASES)
    passed = 0
    failed_cases = []

    # Header
    print("=" * 100)
    print(f"  MEDICINE RECOMMENDATION SYSTEM — TEST SUITE  ({total} Test Cases)")
    print("=" * 100)
    print(f"\n{'#':<4} {'Expected Disease':<38} {'Predicted Disease':<38} {'Sev':>4} {'Result':>8}")
    print("─" * 100)

    for tc in TEST_CASES:
        result = predict(
            tc["symptoms"], rf, le, all_symptoms, sev_dict, desc_dict, prec_dict
        )
        match = result["disease"] == tc["expected"]
        status = "✅ PASS" if match else "❌ FAIL"
        if match:
            passed += 1
        else:
            failed_cases.append(tc)

        print(
            f"{tc['id']:<4} "
            f"{tc['expected']:<38} "
            f"{result['disease']:<38} "
            f"{result['severity_score']:>4} "
            f"{status:>8}"
        )

    # Summary
    print("─" * 100)
    pct = (passed / total) * 100
    print(f"\n  Results: {passed}/{total} passed ({pct:.1f}%)")

    if failed_cases:
        print("\n  ❌ Failed cases:")
        for tc in failed_cases:
            print(f"     Test #{tc['id']}: expected '{tc['expected']}', "
                  f"symptoms = {tc['symptoms']}")
    else:
        print("  🎉 All test cases passed!\n")

    # Detailed output for first 5 cases
    print("\n" + "=" * 100)
    print("  DETAILED OUTPUT — First 5 Test Cases")
    print("=" * 100)

    for tc in TEST_CASES[:5]:
        result = predict(
            tc["symptoms"], rf, le, all_symptoms, sev_dict, desc_dict, prec_dict
        )
        print(f"\n── Test #{tc['id']} ──")
        print(f"  Symptoms       : {', '.join(tc['symptoms'])}")
        print(f"  Predicted      : {result['disease']}")
        print(f"  Severity Score : {result['severity_score']} / 70")
        print(f"  Description    : {result['description'][:150]}...")
        print(f"  Precautions    :")
        for i, p in enumerate(result["precautions"], 1):
            print(f"    {i}. {p}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
