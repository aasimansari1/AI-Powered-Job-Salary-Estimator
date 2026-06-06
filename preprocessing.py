"""
Shared preprocessing utilities for training and prediction.
All categorical lists, encoders, and feature-engineering logic live here
so training and inference always use identical transformations.
"""
import numpy as np
import pandas as pd
from generate_data import (
    JOB_TITLES, EDUCATION_MULTIPLIERS, COUNTRIES_CITIES,
    INDUSTRIES, COMPANY_SIZES, EMPLOYMENT_TYPES, WORK_MODES,
    ALL_SKILLS, ALL_CERTIFICATIONS,
)

# ── Canonical category lists ─────────────────────────────────────────────────

JOB_TITLE_LIST = sorted(JOB_TITLES.keys())
EDUCATION_LIST = [
    'High School / GED',
    "Associate's Degree",
    "Bachelor's Degree",
    "Master's Degree",
    'PhD / Doctorate',
    'Bootcamp Graduate',
]
COUNTRY_LIST = sorted(COUNTRIES_CITIES.keys())
CITY_LIST = sorted({c for cities in COUNTRIES_CITIES.values() for c in cities})
INDUSTRY_LIST = sorted(INDUSTRIES.keys())
COMPANY_SIZE_LIST = [
    'Startup (1-50)',
    'Small (51-200)',
    'Medium (201-1000)',
    'Large (1001-5000)',
    'Enterprise (5000+)',
]
EMPLOYMENT_TYPE_LIST = sorted(EMPLOYMENT_TYPES.keys())
WORK_MODE_LIST = ['On-site', 'Hybrid', 'Remote']
SKILL_LIST = sorted(ALL_SKILLS)
CERT_LIST = sorted(ALL_CERTIFICATIONS)

# Ordinal encodings (preserve ordering)
EDU_ORDER = {e: i for i, e in enumerate(EDUCATION_LIST)}
COMPANY_ORDER = {c: i for i, c in enumerate(COMPANY_SIZE_LIST)}
WORK_ORDER = {w: i for i, w in enumerate(WORK_MODE_LIST)}


def _safe_index(lst: list, val, default: int = 0) -> int:
    try:
        return lst.index(val)
    except ValueError:
        return default


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fill missing values, clamp outliers."""
    df = df.drop_duplicates().copy()
    df['years_experience'] = df['years_experience'].fillna(df['years_experience'].median())
    df['education_level'] = df['education_level'].fillna("Bachelor's Degree")
    df['country'] = df['country'].fillna('United States')
    df['city'] = df['city'].fillna('Unknown')
    df['industry'] = df['industry'].fillna('Technology')
    df['company_size'] = df['company_size'].fillna('Medium (201-1000)')
    df['employment_type'] = df['employment_type'].fillna('Full-time')
    df['work_mode'] = df['work_mode'].fillna('On-site')
    df['skills'] = df['skills'].fillna('')
    df['certifications'] = df['certifications'].fillna('None')

    if 'annual_salary_usd' in df.columns:
        q_low = df['annual_salary_usd'].quantile(0.005)
        q_high = df['annual_salary_usd'].quantile(0.995)
        df = df[(df['annual_salary_usd'] >= q_low) & (df['annual_salary_usd'] <= q_high)]

    return df.reset_index(drop=True)


def encode_row(row: dict) -> np.ndarray:
    """
    Encode a single input dict into a 1-D feature vector.
    Must be called both during training (row-by-row via apply) and inference.
    """
    features = []

    # ── Numerical ────────────────────────────────────────────────────────────
    features.append(float(row.get('years_experience', 0)))

    # ── Ordinal categoricals ──────────────────────────────────────────────────
    edu = str(row.get('education_level', "Bachelor's Degree"))
    features.append(float(EDU_ORDER.get(edu, 2)))

    cs = str(row.get('company_size', 'Medium (201-1000)'))
    features.append(float(COMPANY_ORDER.get(cs, 2)))

    wm = str(row.get('work_mode', 'On-site'))
    features.append(float(WORK_ORDER.get(wm, 0)))

    # ── Nominal categoricals (label-encoded by sorted index) ──────────────────
    features.append(float(_safe_index(JOB_TITLE_LIST, row.get('job_title', ''))))
    features.append(float(_safe_index(COUNTRY_LIST, row.get('country', ''))))
    features.append(float(_safe_index(CITY_LIST, row.get('city', ''))))
    features.append(float(_safe_index(INDUSTRY_LIST, row.get('industry', ''))))
    features.append(float(_safe_index(EMPLOYMENT_TYPE_LIST, row.get('employment_type', ''))))

    # ── Multi-hot skills ─────────────────────────────────────────────────────
    raw_skills = row.get('skills', '')
    if isinstance(raw_skills, list):
        skill_set = set(raw_skills)
    else:
        skill_set = set(str(raw_skills).split(';')) if raw_skills else set()

    for s in SKILL_LIST:
        features.append(1.0 if s in skill_set else 0.0)

    features.append(float(len(skill_set)))  # skill count

    # ── Multi-hot certifications ──────────────────────────────────────────────
    raw_certs = row.get('certifications', 'None')
    if isinstance(raw_certs, list):
        cert_set = set(raw_certs)
    else:
        cert_set = set(str(raw_certs).split(';')) if raw_certs else {'None'}

    for c in CERT_LIST:
        if c == 'None':
            continue
        features.append(1.0 if c in cert_set else 0.0)

    features.append(float(sum(1 for c in cert_set if c != 'None')))

    return np.array(features, dtype=np.float32)


def get_feature_names() -> list:
    """Return ordered list of feature names (mirrors encode_row output)."""
    names = [
        'years_experience',
        'education_level_ord',
        'company_size_ord',
        'work_mode_ord',
        'job_title_enc',
        'country_enc',
        'city_enc',
        'industry_enc',
        'employment_type_enc',
    ]
    names += [f'skill_{s.replace(" ", "_").lower()}' for s in SKILL_LIST]
    names += ['num_skills']
    names += [f'cert_{c.replace(" ", "_").replace("(", "").replace(")", "").lower()}' for c in CERT_LIST if c != 'None']
    names += ['num_certs']
    return names


def prepare_features(df: pd.DataFrame):
    """Transform a cleaned DataFrame → (X, y) ready for sklearn."""
    X = np.vstack(df.apply(lambda r: encode_row(r.to_dict()), axis=1).values)
    y = df['annual_salary_usd'].values.astype(np.float32) if 'annual_salary_usd' in df.columns else None
    return X, y
