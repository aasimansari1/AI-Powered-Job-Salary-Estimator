"""
Prediction utilities.  Loads the saved model and returns a rich result dict
including estimated salary, range, confidence, factor impacts, and career tips.
"""
import os
import json
import numpy as np
import joblib

MODEL_PATH = 'models/best_model.pkl'
RESULTS_PATH = 'models/model_results.json'

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run `python train.py` first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def _get_model_r2() -> float:
    """Return best model R² from saved results (used for confidence estimate)."""
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        return float(data['best']['r2'])
    except Exception:
        return 0.82


def predict_salary(input_dict: dict) -> dict:
    """
    input_dict keys:
        job_title, years_experience, education_level, skills (list or str),
        country, city, industry, company_size, employment_type,
        work_mode, certifications (list or str)

    Returns dict:
        salary, salary_min, salary_max, confidence,
        impacts (experience, skills, location, education),
        growth_projection (list of dicts)
    """
    import pandas as pd
    from preprocessing import encode_row, get_feature_names

    model = load_model()
    row = dict(input_dict)

    def _to_df(r):
        return pd.DataFrame([encode_row(r)], columns=get_feature_names())

    # ── Base prediction ──────────────────────────────────────────────────────
    salary = float(model.predict(_to_df(row))[0])
    salary = max(5000.0, salary)

    r2 = _get_model_r2()
    # Confidence: scale R² into 55–97 range and adjust by residual spread
    confidence = round(55 + r2 * 42, 1)
    confidence = min(97.0, max(55.0, confidence))

    # ── Salary range (±1 std, approximated from R²) ──────────────────────────
    spread_pct = 0.10 + (1 - r2) * 0.12
    salary_min = max(5000.0, salary * (1 - spread_pct))
    salary_max = salary * (1 + spread_pct)

    # ── Factor impact analysis (ablation-style) ──────────────────────────────
    def predict_ablated(**overrides):
        row2 = dict(row)
        row2.update(overrides)
        return float(model.predict(_to_df(row2))[0])

    base_no_exp = predict_ablated(years_experience=0)
    exp_impact = round((salary - base_no_exp) / max(salary, 1) * 100, 1)

    base_no_skills = predict_ablated(skills=[], certifications='None')
    skills_impact = round((salary - base_no_skills) / max(salary, 1) * 100, 1)

    base_us = predict_ablated(country='United States', city='New York')
    loc_impact = round((salary - base_us) / max(base_us, 1) * 100, 1)

    base_bs = predict_ablated(education_level="Bachelor's Degree")
    edu_impact = round((salary - base_bs) / max(base_bs, 1) * 100, 1)

    # ── 5-year growth projection ──────────────────────────────────────────────
    current_yoe = int(input_dict.get('years_experience', 0))
    growth = []
    for delta in range(0, 6):
        proj_row = dict(row)
        proj_row['years_experience'] = current_yoe + delta
        growth.append({
            'year': f'Year +{delta}' if delta > 0 else 'Now',
            'years_exp': current_yoe + delta,
            'salary': round(float(model.predict(_to_df(proj_row))[0])),
        })

    # ── High-paying skill recommendations ────────────────────────────────────
    from generate_data import SKILL_PREMIUMS, ALL_SKILLS
    raw_skills = input_dict.get('skills', [])
    if isinstance(raw_skills, str):
        current_skills = set(raw_skills.split(';')) if raw_skills else set()
    else:
        current_skills = set(raw_skills)

    missing_skills = {s: SKILL_PREMIUMS[s] for s in ALL_SKILLS if s not in current_skills}
    top_skills = sorted(missing_skills.items(), key=lambda x: x[1], reverse=True)[:8]

    return {
        'salary': round(salary),
        'salary_min': round(salary_min),
        'salary_max': round(salary_max),
        'confidence': confidence,
        'impacts': {
            'experience': exp_impact,
            'skills': max(0.0, skills_impact),
            'location': loc_impact,
            'education': edu_impact,
        },
        'growth_projection': growth,
        'recommended_skills': [s for s, _ in top_skills],
    }


def get_career_paths(job_title: str) -> list:
    """Return a simple career progression ladder for the given role."""
    paths = {
        'Software Engineer': [
            'Junior Software Engineer → Software Engineer → Senior Software Engineer',
            'Senior Software Engineer → Technical Lead → Engineering Manager',
            'Senior Software Engineer → Solutions Architect → Cloud Architect',
        ],
        'Data Scientist': [
            'Data Analyst → Data Scientist → Senior Data Scientist',
            'Senior Data Scientist → ML Engineer → AI/ML Researcher',
            'Senior Data Scientist → Data Science Manager → VP of Data',
        ],
        'Machine Learning Engineer': [
            'Data Scientist → ML Engineer → Senior ML Engineer',
            'Senior ML Engineer → AI/ML Researcher → Principal Scientist',
            'Senior ML Engineer → ML Platform Lead → Engineering Manager',
        ],
        'Data Analyst': [
            'Data Analyst → Senior Data Analyst → Analytics Manager',
            'Senior Data Analyst → Data Scientist → ML Engineer',
            'Analytics Manager → Director of Analytics → Chief Data Officer',
        ],
        'DevOps Engineer': [
            'DevOps Engineer → Senior DevOps → Site Reliability Engineer',
            'Senior SRE → Cloud Architect → Principal Architect',
            'SRE → Engineering Manager → VP of Infrastructure',
        ],
        'Product Manager': [
            'Associate PM → Product Manager → Senior PM',
            'Senior PM → Principal PM → VP of Product',
            'Senior PM → Director of Product → Chief Product Officer',
        ],
    }
    default = [
        f'Junior {job_title} → {job_title} → Senior {job_title}',
        f'Senior {job_title} → Lead {job_title} → Manager',
        f'Manager → Director → VP / C-Suite',
    ]
    return paths.get(job_title, default)
