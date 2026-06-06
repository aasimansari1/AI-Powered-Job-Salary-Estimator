"""
Generates a synthetic, realistic job salary dataset for training the ML model.
Uses carefully calibrated multipliers to produce plausible salary distributions.
"""
import numpy as np
import pandas as pd
import os
import random

np.random.seed(42)
random.seed(42)

JOB_TITLES = {
    'Software Engineer': 110000,
    'Senior Software Engineer': 148000,
    'Data Scientist': 122000,
    'Machine Learning Engineer': 138000,
    'Data Analyst': 76000,
    'Senior Data Analyst': 102000,
    'Product Manager': 128000,
    'Frontend Developer': 97000,
    'Backend Developer': 110000,
    'Full Stack Developer': 107000,
    'DevOps Engineer': 118000,
    'Cloud Architect': 155000,
    'Security Engineer': 122000,
    'Data Engineer': 120000,
    'AI/ML Researcher': 160000,
    'Mobile Developer': 110000,
    'UX/UI Designer': 92000,
    'Business Analyst': 86000,
    'Project Manager': 102000,
    'Database Administrator': 94000,
    'Blockchain Developer': 132000,
    'Site Reliability Engineer': 133000,
    'QA Engineer': 80000,
    'Network Engineer': 92000,
    'Systems Architect': 148000,
    'Cybersecurity Analyst': 108000,
    'Solutions Architect': 152000,
    'Technical Lead': 145000,
    'Engineering Manager': 162000,
    'Data Science Manager': 168000,
}

EDUCATION_MULTIPLIERS = {
    'High School / GED': 0.78,
    "Associate's Degree": 0.88,
    "Bachelor's Degree": 1.00,
    "Master's Degree": 1.16,
    'PhD / Doctorate': 1.30,
    'Bootcamp Graduate': 0.91,
}

COUNTRIES_CITIES = {
    'United States': ['San Francisco', 'New York', 'Seattle', 'Austin', 'Boston', 'Chicago', 'Los Angeles', 'Denver', 'Atlanta', 'Miami'],
    'United Kingdom': ['London', 'Manchester', 'Birmingham', 'Edinburgh', 'Bristol', 'Leeds'],
    'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Ottawa', 'Calgary'],
    'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
    'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
    'India': ['Bengaluru', 'Hyderabad', 'Mumbai', 'Pune', 'Delhi', 'Chennai'],
    'Singapore': ['Singapore City'],
    'Netherlands': ['Amsterdam', 'Rotterdam', 'The Hague'],
    'Switzerland': ['Zurich', 'Geneva', 'Basel'],
    'France': ['Paris', 'Lyon', 'Marseille'],
    'Japan': ['Tokyo', 'Osaka', 'Yokohama'],
    'Brazil': ['São Paulo', 'Rio de Janeiro', 'Curitiba'],
    'Poland': ['Warsaw', 'Kraków', 'Wrocław'],
    'UAE': ['Dubai', 'Abu Dhabi'],
    'South Korea': ['Seoul', 'Busan'],
    'Mexico': ['Mexico City', 'Monterrey'],
    'China': ['Beijing', 'Shanghai', 'Shenzhen'],
    'Sweden': ['Stockholm', 'Gothenburg'],
    'South Africa': ['Cape Town', 'Johannesburg'],
    'Argentina': ['Buenos Aires', 'Córdoba'],
}

LOCATION_MULTIPLIERS = {
    'United States': 1.00,
    'Switzerland': 1.06,
    'Australia': 0.83,
    'Canada': 0.79,
    'United Kingdom': 0.84,
    'Germany': 0.73,
    'Netherlands': 0.76,
    'Singapore': 0.89,
    'Japan': 0.69,
    'France': 0.71,
    'South Korea': 0.63,
    'UAE': 0.83,
    'Sweden': 0.78,
    'India': 0.19,
    'Brazil': 0.29,
    'Mexico': 0.26,
    'Poland': 0.41,
    'China': 0.39,
    'South Africa': 0.31,
    'Argentina': 0.26,
}

INDUSTRIES = {
    'Technology': 1.13,
    'Finance & Banking': 1.20,
    'Healthcare & Pharma': 1.06,
    'E-commerce & Retail': 1.09,
    'Consulting': 1.11,
    'Government & Public Sector': 0.86,
    'Education & Research': 0.81,
    'Manufacturing': 0.91,
    'Media & Entertainment': 0.94,
    'Energy & Utilities': 1.06,
    'Telecommunications': 1.01,
    'Insurance': 1.03,
    'Aerospace & Defense': 1.09,
    'Automotive': 0.96,
    'Startups': 1.06,
}

COMPANY_SIZES = {
    'Startup (1–50)': 0.88,
    'Small (51–200)': 0.94,
    'Medium (201–1000)': 1.00,
    'Large (1001–5000)': 1.11,
    'Enterprise (5000+)': 1.20,
}

EMPLOYMENT_TYPES = {
    'Full-time': 1.00,
    'Part-time': 0.55,
    'Contract': 1.19,
    'Freelance': 1.09,
    'Internship': 0.38,
}

WORK_MODES = {
    'On-site': 1.00,
    'Hybrid': 1.04,
    'Remote': 1.07,
}

ALL_SKILLS = [
    'Python', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow',
    'PyTorch', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'React',
    'Node.js', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go', 'Rust',
    'Scala', 'Apache Spark', 'Kafka', 'MongoDB', 'PostgreSQL', 'Redis',
    'GraphQL', 'REST APIs', 'CI/CD', 'Git', 'Linux',
]

SKILL_PREMIUMS = {
    'Python': 0.055, 'SQL': 0.030, 'Machine Learning': 0.110, 'Deep Learning': 0.130,
    'TensorFlow': 0.075, 'PyTorch': 0.085, 'AWS': 0.085, 'Azure': 0.075,
    'GCP': 0.065, 'Docker': 0.055, 'Kubernetes': 0.090, 'React': 0.055,
    'Node.js': 0.045, 'JavaScript': 0.040, 'TypeScript': 0.055, 'Java': 0.045,
    'C++': 0.065, 'Go': 0.085, 'Rust': 0.105, 'Scala': 0.090,
    'Apache Spark': 0.095, 'Kafka': 0.088, 'MongoDB': 0.042, 'PostgreSQL': 0.038,
    'Redis': 0.048, 'GraphQL': 0.052, 'REST APIs': 0.035, 'CI/CD': 0.045,
    'Git': 0.020, 'Linux': 0.040,
}

ALL_CERTIFICATIONS = [
    'AWS Certified Solutions Architect',
    'Google Cloud Professional',
    'Azure Solutions Architect',
    'PMP (Project Management)',
    'CFA (Finance)',
    'CISSP (Security)',
    'CKA (Kubernetes)',
    'Salesforce Certified',
    'Data Science Professional (IBM)',
    'CompTIA Security+',
    'None',
]

CERT_PREMIUMS = {
    'AWS Certified Solutions Architect': 0.060,
    'Google Cloud Professional': 0.058,
    'Azure Solutions Architect': 0.055,
    'PMP (Project Management)': 0.065,
    'CFA (Finance)': 0.085,
    'CISSP (Security)': 0.075,
    'CKA (Kubernetes)': 0.065,
    'Salesforce Certified': 0.052,
    'Data Science Professional (IBM)': 0.045,
    'CompTIA Security+': 0.040,
    'None': 0.000,
}

ROLE_TYPICAL_SKILLS = {
    'Software Engineer': ['Python', 'Java', 'C++', 'Git', 'SQL', 'Docker', 'REST APIs'],
    'Senior Software Engineer': ['Python', 'Java', 'C++', 'Git', 'Docker', 'Kubernetes', 'CI/CD'],
    'Data Scientist': ['Python', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Apache Spark'],
    'Machine Learning Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'Docker', 'Kubernetes', 'AWS'],
    'Data Analyst': ['Python', 'SQL', 'PostgreSQL', 'Git', 'MongoDB'],
    'Senior Data Analyst': ['Python', 'SQL', 'PostgreSQL', 'Machine Learning', 'Apache Spark'],
    'Product Manager': ['SQL', 'REST APIs', 'Git'],
    'Frontend Developer': ['JavaScript', 'TypeScript', 'React', 'GraphQL', 'Git'],
    'Backend Developer': ['Python', 'Java', 'Node.js', 'SQL', 'PostgreSQL', 'Redis', 'Docker'],
    'Full Stack Developer': ['JavaScript', 'TypeScript', 'React', 'Node.js', 'SQL', 'MongoDB', 'Docker'],
    'DevOps Engineer': ['Docker', 'Kubernetes', 'AWS', 'CI/CD', 'Linux', 'Python'],
    'Cloud Architect': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'Terraform'],
    'Security Engineer': ['Linux', 'Python', 'AWS', 'CI/CD', 'Docker'],
    'Data Engineer': ['Python', 'SQL', 'Apache Spark', 'Kafka', 'AWS', 'PostgreSQL', 'MongoDB'],
    'AI/ML Researcher': ['Python', 'PyTorch', 'TensorFlow', 'Deep Learning', 'Machine Learning'],
    'Mobile Developer': ['Java', 'JavaScript', 'TypeScript', 'React', 'Git'],
    'UX/UI Designer': ['JavaScript', 'TypeScript', 'React'],
    'Business Analyst': ['SQL', 'Python', 'REST APIs'],
    'Project Manager': ['SQL', 'Git'],
    'Database Administrator': ['SQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Linux'],
    'Blockchain Developer': ['Rust', 'Go', 'Python', 'SQL'],
    'Site Reliability Engineer': ['Docker', 'Kubernetes', 'AWS', 'Python', 'Linux', 'CI/CD'],
    'QA Engineer': ['Python', 'SQL', 'Git', 'CI/CD', 'REST APIs'],
    'Network Engineer': ['Linux', 'Python', 'AWS', 'Azure'],
    'Systems Architect': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'Linux'],
    'Cybersecurity Analyst': ['Linux', 'Python', 'AWS', 'CI/CD'],
    'Solutions Architect': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'REST APIs'],
    'Technical Lead': ['Python', 'Java', 'Docker', 'Kubernetes', 'CI/CD', 'Git'],
    'Engineering Manager': ['Python', 'Docker', 'Kubernetes', 'CI/CD', 'AWS'],
    'Data Science Manager': ['Python', 'Machine Learning', 'SQL', 'Apache Spark'],
}


def _exp_multiplier(years: float) -> float:
    return 0.68 + 0.040 * min(float(years), 22)


def _skills_premium(skills_list: list) -> float:
    total = sum(SKILL_PREMIUMS.get(s, 0.0) for s in skills_list)
    return 1.0 + min(total, 0.38)


def _certs_premium(certs_list: list) -> float:
    total = sum(CERT_PREMIUMS.get(c, 0.0) for c in certs_list if c != 'None')
    return 1.0 + min(total, 0.22)


def _random_skills(job_title: str, n_extra: int = 0) -> list:
    base = ROLE_TYPICAL_SKILLS.get(job_title, ALL_SKILLS[:5])
    pool = [s for s in ALL_SKILLS if s not in base]
    extras = random.sample(pool, min(n_extra, len(pool)))
    chosen = list(set(base + extras))
    k = random.randint(max(1, len(base) - 2), min(len(base) + 3, len(chosen)))
    return random.sample(chosen, k)


def generate_salary_data(n_samples: int = 15000) -> pd.DataFrame:
    """Generate a synthetic salary dataset with realistic distributions."""
    np.random.seed(42)
    random.seed(42)

    rows = []
    countries = list(COUNTRIES_CITIES.keys())
    country_weights = [
        0.32, 0.07, 0.07, 0.06, 0.06,
        0.12, 0.03, 0.04, 0.03, 0.04,
        0.03, 0.02, 0.02, 0.02, 0.02,
        0.01, 0.01, 0.01, 0.01, 0.01,
    ]

    job_titles = list(JOB_TITLES.keys())
    educations = list(EDUCATION_MULTIPLIERS.keys())
    edu_weights = [0.06, 0.08, 0.45, 0.28, 0.08, 0.05]
    industries = list(INDUSTRIES.keys())
    company_sizes = list(COMPANY_SIZES.keys())
    employment_types = list(EMPLOYMENT_TYPES.keys())
    emp_weights = [0.62, 0.08, 0.15, 0.10, 0.05]
    work_modes = list(WORK_MODES.keys())

    for _ in range(n_samples):
        job_title = random.choice(job_titles)
        base = JOB_TITLES[job_title]

        # Experience weighted toward mid-career
        years_exp = max(0, int(np.random.lognormal(mean=1.8, sigma=0.8)))
        years_exp = min(years_exp, 35)

        education = random.choices(educations, weights=edu_weights)[0]
        country = random.choices(countries, weights=country_weights)[0]
        city = random.choice(COUNTRIES_CITIES[country])
        industry = random.choice(industries)
        company_size = random.choice(company_sizes)
        employment_type = random.choices(employment_types, weights=emp_weights)[0]
        work_mode = random.choice(work_modes)

        n_extra = random.randint(0, 4)
        skills = _random_skills(job_title, n_extra)

        certs = random.choices(
            ALL_CERTIFICATIONS,
            weights=[0.08, 0.07, 0.07, 0.06, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.40]
        )
        # 10% chance of 2 certifications
        if random.random() < 0.10 and certs[0] != 'None':
            extra_cert = random.choice([c for c in ALL_CERTIFICATIONS if c != certs[0] and c != 'None'])
            certs = [certs[0], extra_cert]
        else:
            certs = [certs[0]]

        salary = (
            base
            * _exp_multiplier(years_exp)
            * EDUCATION_MULTIPLIERS[education]
            * LOCATION_MULTIPLIERS[country]
            * INDUSTRIES[industry]
            * COMPANY_SIZES[company_size]
            * EMPLOYMENT_TYPES[employment_type]
            * WORK_MODES[work_mode]
            * _skills_premium(skills)
            * _certs_premium(certs)
        )

        # Gaussian noise ±9%
        salary *= np.random.normal(1.0, 0.09)

        # Small fraction of outliers (top earners / anomalies)
        if random.random() < 0.03:
            salary *= random.uniform(1.25, 1.75)

        salary = max(8000, round(salary, -2))

        rows.append({
            'job_title': job_title,
            'years_experience': years_exp,
            'education_level': education,
            'skills': ';'.join(skills),
            'country': country,
            'city': city,
            'industry': industry,
            'company_size': company_size,
            'employment_type': employment_type,
            'work_mode': work_mode,
            'certifications': ';'.join(certs),
            'annual_salary_usd': int(salary),
        })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} salary records.")
    print(f"Salary range: ${df['annual_salary_usd'].min():,} – ${df['annual_salary_usd'].max():,}")
    print(f"Median salary: ${df['annual_salary_usd'].median():,.0f}")
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_salary_data(15000)
    df.to_csv('data/salary_data.csv', index=False)
    print(f"\nSaved to data/salary_data.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(df.head())
