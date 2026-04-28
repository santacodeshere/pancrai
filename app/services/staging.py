"""
PancrAI — TNM Staging and Clinical Risk Score Calculator

TNM Staging for Pancreatic Cancer (AJCC 8th Edition):
    T1: Tumor ≤ 2cm confined to pancreas
    T2: Tumor > 2cm ≤ 4cm confined to pancreas
    T3: Tumor > 4cm OR involving duodenum/bile duct/peripancreatic tissues
    T4: Tumor involves major vessels (celiac, SMA, hepatic artery)

Resectability Assessment:
    Resectable    : No vascular involvement, no distant metastasis
    Borderline    : Abutment of major vessels < 180°
    Unresectable  : Encasement of major vessels > 180° OR distant metastasis

Composite Risk Score (0-100):
    Combines tumor size, morphology, uncertainty, and classification
"""

import numpy as np
from typing import Dict, Optional


# ── TNM T-Stage ────────────────────────────────────────────────────────────────

def get_t_stage(area_cm2: float, tumor_class: str) -> Dict:
    """
    Estimate T-stage from tumor area measurement.
    Note: This is an approximation — true T-staging requires 3D volume
    and vascular assessment from full volumetric CT review.

    Args:
        area_cm2: Tumor cross-sectional area in cm²
        tumor_class: Classifier output class name

    Returns:
        Dict with stage, description, size_estimate
    """
    if tumor_class == "No Tumor" or area_cm2 < 0.01:
        return {
            "t_stage": "T0",
            "description": "No evidence of primary tumor",
            "size_estimate": "< 0.1 cm",
            "color": "#4CAF50",
        }

    # Estimate diameter from area (assume circular cross-section)
    diameter_cm = 2 * np.sqrt(area_cm2 / np.pi)

    if diameter_cm <= 2.0:
        stage, desc, color = "T1", "Tumor ≤ 2cm, confined to pancreas", "#FF9800"
        sub = "T1a" if diameter_cm <= 0.5 else "T1b" if diameter_cm <= 1.0 else "T1c"
        stage = sub
    elif diameter_cm <= 4.0:
        stage, desc, color = "T2", "Tumor > 2cm ≤ 4cm, confined to pancreas", "#FF6B35"
    else:
        stage, desc, color = "T3", "Tumor > 4cm or involving adjacent structures", "#F44336"

    if tumor_class == "Malignant (PDAC)" and area_cm2 > 5.0:
        stage, desc, color = "T4", "Tumor involves major vessels or adjacent organs", "#B71C1C"

    return {
        "t_stage":       stage,
        "description":   desc,
        "size_estimate": f"~{diameter_cm:.1f} cm diameter",
        "area_cm2":      round(area_cm2, 3),
        "color":         color,
    }


def get_overall_stage(t_stage: str, tumor_class: str) -> Dict:
    """
    Estimate overall clinical stage from T-stage and tumor class.
    N and M stages are assumed N0M0 (no lymph node/metastasis data from imaging alone).

    Returns:
        Dict with stage, survival_rate, prognosis
    """
    if t_stage == "T0":
        return {
            "overall_stage": "Stage 0",
            "tnm":           "T0 N0 M0",
            "five_year_survival": "N/A",
            "prognosis":     "No tumor detected",
            "color":         "#4CAF50",
        }

    if tumor_class == "No Tumor":
        return {
            "overall_stage": "Stage 0",
            "tnm":           "T0 N0 M0",
            "five_year_survival": "N/A",
            "prognosis":     "No tumor detected",
            "color":         "#4CAF50",
        }

    if tumor_class == "Benign":
        return {
            "overall_stage": "Benign",
            "tnm":           f"{t_stage} N0 M0",
            "five_year_survival": "> 90%",
            "prognosis":     "Generally favorable with appropriate management",
            "color":         "#FF9800",
        }

    if tumor_class == "Cystic (IPMN)":
        return {
            "overall_stage": "IPMN",
            "tnm":           f"{t_stage} N0 M0",
            "five_year_survival": "60-80% (depends on subtype)",
            "prognosis":     "Variable — requires surveillance; main duct IPMN has higher malignant potential",
            "color":         "#2196F3",
        }

    # Malignant PDAC staging
    stage_map = {
        "T1a": ("Stage IA",  "63-80%",  "Excellent — small resectable tumor"),
        "T1b": ("Stage IA",  "63-80%",  "Excellent — small resectable tumor"),
        "T1c": ("Stage IB",  "40-63%",  "Good — resectable, adjuvant therapy recommended"),
        "T2":  ("Stage IIA", "20-40%",  "Moderate — surgical resection typically possible"),
        "T3":  ("Stage IIB", "10-20%",  "Limited — borderline resectability assessment required"),
        "T4":  ("Stage III", "3-10%",   "Poor — locally advanced, typically unresectable"),
    }

    stage_data = stage_map.get(t_stage, ("Stage II", "15-30%", "Requires multidisciplinary review"))
    colors = {
        "Stage IA": "#FF9800", "Stage IB": "#FF6B35",
        "Stage IIA": "#F44336", "Stage IIB": "#E53935",
        "Stage III": "#B71C1C", "Stage IV": "#7B1FA2",
    }

    return {
        "overall_stage":       stage_data[0],
        "tnm":                 f"{t_stage} N0 M0 (estimated)",
        "five_year_survival":  stage_data[1],
        "prognosis":           stage_data[2],
        "color":               colors.get(stage_data[0], "#F44336"),
        "note":                "N and M staging require lymph node biopsy and systemic imaging",
    }


# ── Resectability ──────────────────────────────────────────────────────────────

def assess_resectability(t_stage: str, tumor_class: str,
                          area_cm2: float) -> Dict:
    """
    Estimate surgical resectability based on T-stage and tumor characteristics.
    True resectability requires CT angiography and MDT assessment.
    """
    if tumor_class == "No Tumor":
        return {"status": "N/A", "description": "No tumor detected", "color": "#4CAF50"}

    if tumor_class == "Benign":
        return {
            "status":      "Likely Resectable",
            "description": "Benign lesions are typically surgically accessible",
            "procedure":   "Enucleation or distal pancreatectomy depending on location",
            "color":       "#4CAF50",
        }

    if tumor_class == "Cystic (IPMN)":
        return {
            "status":      "Resectable (if symptomatic or high-risk features)",
            "description": "IPMN management follows Fukuoka guidelines",
            "procedure":   "Partial pancreatectomy with negative margins",
            "color":       "#FF9800",
        }

    # PDAC resectability
    if t_stage in ["T1a", "T1b", "T1c", "T2"]:
        return {
            "status":      "Resectable",
            "description": "Tumor appears confined to pancreas without major vascular involvement",
            "procedure":   "Whipple procedure (pancreaticoduodenectomy) or distal pancreatectomy",
            "color":       "#4CAF50",
        }
    elif t_stage == "T3":
        return {
            "status":      "Borderline Resectable",
            "description": "Possible peripancreatic tissue involvement — requires CT angiography",
            "procedure":   "Neoadjuvant chemotherapy followed by surgical reassessment",
            "color":       "#FF9800",
        }
    else:
        return {
            "status":      "Locally Advanced / Unresectable",
            "description": "Suspected major vascular involvement — systemic therapy indicated",
            "procedure":   "FOLFIRINOX or gemcitabine/nab-paclitaxel chemotherapy",
            "color":       "#F44336",
        }


# ── Composite Risk Score ───────────────────────────────────────────────────────

def calculate_risk_score(
    tumor_class:      str,
    confidence:       float,
    area_cm2:         float,
    uncertainty_score: float,
    solidity:         float = 1.0,
    circularity:      float = 1.0,
    patient_age:      Optional[int] = None,
) -> Dict:
    """
    Calculate composite clinical risk score (0-100).

    Components:
        - Tumor class weight (0-40 pts)
        - Tumor size (0-20 pts)
        - Model confidence (0-15 pts)
        - Morphological irregularity (0-15 pts)
        - Uncertainty penalty (0-10 pts)

    Args:
        tumor_class: Classification result
        confidence: Model confidence (0-1)
        area_cm2: Tumor area in cm²
        uncertainty_score: MC Dropout uncertainty (0-100)
        solidity: Mask solidity (0-1)
        circularity: Mask circularity (0-1)
        patient_age: Optional patient age for age-adjusted risk

    Returns:
        Dict with score, category, components, recommendations
    """
    score = 0.0
    components = {}

    # Component 1: Tumor class (0-40 pts)
    class_scores = {
        "No Tumor":         0,
        "Benign":          15,
        "Cystic (IPMN)":   25,
        "Malignant (PDAC)":40,
    }
    class_pts = class_scores.get(tumor_class, 20) * confidence
    score += class_pts
    components["Tumor Classification"] = round(class_pts, 1)

    # Component 2: Tumor size (0-20 pts)
    if area_cm2 > 0:
        diameter = 2 * np.sqrt(area_cm2 / np.pi)
        size_pts = min(20, diameter * 5)  # 4cm+ = max 20pts
    else:
        size_pts = 0
    score += size_pts
    components["Tumor Size"] = round(size_pts, 1)

    # Component 3: Confidence-weighted certainty (0-15 pts)
    conf_pts = confidence * 15
    score += conf_pts
    components["Classification Confidence"] = round(conf_pts, 1)

    # Component 4: Morphological irregularity (0-15 pts)
    if tumor_class != "No Tumor" and area_cm2 > 0:
        irregularity = (1 - solidity) * 0.6 + (1 - circularity) * 0.4
        morph_pts = irregularity * 15
        score += morph_pts
        components["Morphological Irregularity"] = round(morph_pts, 1)
    else:
        components["Morphological Irregularity"] = 0.0

    # Component 5: Uncertainty penalty (0-10 pts)
    unc_pts = (uncertainty_score / 100) * 10
    score += unc_pts
    components["Model Uncertainty"] = round(unc_pts, 1)

    # Age adjustment (optional)
    if patient_age and patient_age > 70 and tumor_class != "No Tumor":
        score = min(100, score * 1.05)
        components["Age Adjustment"] = "+5% (age > 70)"

    score = round(min(100, max(0, score)), 1)

    # Risk category
    if score < 15:
        category, color = "Very Low Risk",  "#4CAF50"
    elif score < 30:
        category, color = "Low Risk",       "#8BC34A"
    elif score < 50:
        category, color = "Moderate Risk",  "#FF9800"
    elif score < 70:
        category, color = "High Risk",      "#FF5722"
    else:
        category, color = "Critical Risk",  "#F44336"

    # Recommendations
    recommendations = _get_recommendations(score, tumor_class, area_cm2, uncertainty_score)

    return {
        "score":           score,
        "category":        category,
        "color":           color,
        "components":      components,
        "recommendations": recommendations,
        "max_score":       100,
    }


def _get_recommendations(score: float, tumor_class: str,
                          area_cm2: float, uncertainty: float) -> list:
    """Generate clinical recommendations based on risk profile."""
    recs = []

    if uncertainty > 60:
        recs.append("⚠️ High model uncertainty — specialist radiologist review strongly recommended")

    if tumor_class == "No Tumor":
        recs.append("✅ No tumor detected — routine follow-up as per clinical guidelines")
        recs.append("📋 Annual surveillance CT if high-risk patient (familial history, genetic syndrome)")

    elif tumor_class == "Benign":
        recs.append("📊 Correlate with clinical symptoms and CA 19-9 serum marker")
        recs.append("🔬 Consider EUS (Endoscopic Ultrasound) for further characterization")
        recs.append("📅 Follow-up imaging in 6 months to assess stability")

    elif tumor_class == "Cystic (IPMN)":
        recs.append("📋 Apply Fukuoka Guidelines for IPMN management")
        recs.append("🔬 MRCP recommended for ductal communication assessment")
        recs.append("🩸 CA 19-9 and CEA serum markers")
        recs.append("👥 Multidisciplinary team (MDT) discussion")
        if area_cm2 > 3:
            recs.append("⚕️ Surgical consultation — size exceeds surveillance threshold")

    elif tumor_class == "Malignant (PDAC)":
        recs.append("🚨 Urgent multidisciplinary oncology team referral")
        recs.append("🩸 CA 19-9, CEA, liver function tests, complete blood count")
        recs.append("🔬 EUS-guided fine needle aspiration (FNA) biopsy for tissue confirmation")
        recs.append("🏥 Staging CT chest/abdomen/pelvis with contrast")
        recs.append("⚕️ Surgical oncology and hepatobiliary surgery consultation")
        recs.append("💊 Discussion of neoadjuvant therapy protocol if borderline resectable")

    if score > 70 and tumor_class != "No Tumor":
        recs.append("🔴 HIGH PRIORITY: Expedited clinical pathway recommended")

    return recs


# ── Full staging report ────────────────────────────────────────────────────────

def generate_staging_report(
    tumor_class:       str,
    confidence:        float,
    measurements:      Optional[Dict],
    uncertainty_score: float,
    patient_age:       Optional[int] = None,
    radiomics:         Optional[Dict] = None,
) -> Dict:
    """
    Generate complete TNM staging and risk assessment report.

    Args:
        tumor_class: Classification result
        confidence: Model confidence (0-1)
        measurements: Dict from segmentation (area_cm2, etc.)
        uncertainty_score: MC Dropout uncertainty (0-100)
        patient_age: Patient age in years
        radiomics: Radiomics feature dict (optional)

    Returns:
        Complete staging report dict
    """
    area_cm2 = measurements.get("area_cm2", 0.0) if measurements else 0.0
    solidity = 1.0
    circularity = 1.0

    if radiomics:
        solidity    = radiomics.get("shape_solidity", 1.0)
        circularity = radiomics.get("shape_circularity", 1.0)

    t_info         = get_t_stage(area_cm2, tumor_class)
    stage_info     = get_overall_stage(t_info["t_stage"], tumor_class)
    resectability  = assess_resectability(t_info["t_stage"], tumor_class, area_cm2)
    risk_score     = calculate_risk_score(
        tumor_class, confidence, area_cm2, uncertainty_score,
        solidity, circularity, patient_age
    )

    return {
        "t_stage":        t_info,
        "overall_stage":  stage_info,
        "resectability":  resectability,
        "risk_score":     risk_score,
        "tumor_class":    tumor_class,
        "area_cm2":       area_cm2,
        "disclaimer":     (
            "This staging assessment is generated by an AI system for "
            "decision support purposes only. TNM staging requires comprehensive "
            "clinical evaluation, multidisciplinary review, and pathological "
            "confirmation. This output must not be used as the sole basis for "
            "clinical decisions."
        ),
    }


if __name__ == "__main__":
    print("Testing TNM staging...\n")

    report = generate_staging_report(
        tumor_class="Malignant (PDAC)",
        confidence=0.87,
        measurements={"area_cm2": 4.2},
        uncertainty_score=25.0,
        patient_age=68,
    )

    print(f"T-Stage: {report['t_stage']['t_stage']} — {report['t_stage']['description']}")
    print(f"Overall Stage: {report['overall_stage']['overall_stage']}")
    print(f"TNM: {report['overall_stage']['tnm']}")
    print(f"5-Year Survival: {report['overall_stage']['five_year_survival']}")
    print(f"Resectability: {report['resectability']['status']}")
    print(f"Risk Score: {report['risk_score']['score']}/100 — {report['risk_score']['category']}")
    print(f"\nRecommendations:")
    for r in report['risk_score']['recommendations']:
        print(f"  {r}")
