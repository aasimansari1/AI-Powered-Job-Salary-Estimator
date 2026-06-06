"""
AI-Powered Job Salary Estimator  -  Streamlit application
Run:  streamlit run app.py
"""
import os
import json
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings('ignore')

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Job Salary Estimator',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradient header */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4facfe, #00f2fe, #43e97b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero-header p { color: #a8c0cc; font-size: 1.05rem; margin: 0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1a2332, #1e2a3d);
    border: 1px solid rgba(79,172,254,0.2);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .lbl {
    font-size: 0.82rem;
    color: #7a8fa6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Salary result banner */
.salary-banner {
    background: linear-gradient(135deg, #0f2027, #1a3a4f);
    border: 2px solid rgba(79,172,254,0.35);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.salary-banner .amount {
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #43e97b, #38f9d7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.salary-banner .range {
    font-size: 1.1rem;
    color: #8ab4c8;
    margin-top: 0.5rem;
}

/* Impact bar */
.impact-row { margin: 0.6rem 0; }
.impact-label { font-size: 0.9rem; color: #c5d8e8; }
.impact-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.impact-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}

/* Skill chip */
.skill-chip {
    display: inline-block;
    background: rgba(79,172,254,0.15);
    border: 1px solid rgba(79,172,254,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    margin: 0.2rem;
    font-size: 0.82rem;
    color: #7dd3fc;
}

/* Section title */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #c8d8e8;
    border-left: 4px solid #4facfe;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Info card */
.info-card {
    background: rgba(79,172,254,0.07);
    border: 1px solid rgba(79,172,254,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0f2236 100%) !important;
}

/* Tab styling */
.stTabs [role="tablist"] { gap: 6px; }
.stTabs [role="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Plotly charts dark bg */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Footer */
.footer {
    text-align: center;
    color: #4a6275;
    font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Constants (imported lazily to avoid circular imports before packages load) ─

@st.cache_data
def get_constants():
    from preprocessing import (
        JOB_TITLE_LIST, EDUCATION_LIST, COUNTRY_LIST, INDUSTRY_LIST,
        COMPANY_SIZE_LIST, EMPLOYMENT_TYPE_LIST, WORK_MODE_LIST,
        SKILL_LIST, CERT_LIST,
    )
    from generate_data import COUNTRIES_CITIES
    return dict(
        job_titles=JOB_TITLE_LIST,
        educations=EDUCATION_LIST,
        countries=COUNTRY_LIST,
        countries_cities=COUNTRIES_CITIES,
        industries=INDUSTRY_LIST,
        company_sizes=COMPANY_SIZE_LIST,
        employment_types=EMPLOYMENT_TYPE_LIST,
        work_modes=WORK_MODE_LIST,
        skills=SKILL_LIST,
        certs=[c for c in sorted(CERT_LIST) if c != 'None'],
    )


# ── Model loading / auto-training ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    import joblib
    if not os.path.exists('models/best_model.pkl'):
        return None
    return joblib.load('models/best_model.pkl')


@st.cache_data(show_spinner=False)
def get_training_results():
    path = 'models/model_results.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataset():
    path = 'data/salary_data.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def run_initialization():
    """
    Runs automatically on cold start when the trained model is absent.
    Generates data (if needed) then trains all models, showing live progress.
    """
    st.markdown("""
    <div style="text-align:center;padding:2rem 1rem 1rem 1rem;">
        <div style="font-size:3.5rem;margin-bottom:0.6rem;">🚀</div>
        <h3 style="color:#4facfe;margin:0 0 0.4rem;">Preparing Your Salary Estimator</h3>
        <p style="color:#7a8fa6;margin:0;font-size:0.95rem;">
            Training 5 ML models on 15,000 salary records - takes about 45-60 seconds.<br>
            This only happens once per session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 3, 1])
    with col_m:
        pbar = st.progress(0, text='📂  Loading salary dataset…')
        time.sleep(0.3)

        os.makedirs('models', exist_ok=True)

        if not os.path.exists('data/salary_data.csv'):
            pbar.progress(5, text='📊  Generating synthetic salary dataset…')
            from generate_data import generate_salary_data
            os.makedirs('data', exist_ok=True)
            df = generate_salary_data(15000)
            df.to_csv('data/salary_data.csv', index=False)

        pbar.progress(15, text='⚙️  Preprocessing features…')
        time.sleep(0.2)
        pbar.progress(25, text='🔧  Training Ridge Regression…')

        import train as tr
        tr.main()

        pbar.progress(100, text='✅  All models trained! Loading app…')
        time.sleep(0.8)

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(C: dict) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.5rem 0;">
            <div style="font-size:2.5rem;">💼</div>
            <div style="font-size:1.1rem;font-weight:700;color:#4facfe;">Salary Estimator</div>
            <div style="font-size:0.75rem;color:#5a7a90;margin-top:0.2rem;">AI-Powered · No Sign-up Required</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="color:#4facfe;font-weight:600;font-size:0.9rem;margin-bottom:0.5rem;">👤 ROLE DETAILS</p>', unsafe_allow_html=True)
        job_title = st.selectbox('Job Title', C['job_titles'], index=C['job_titles'].index('Software Engineer'))
        years_exp = st.slider('Years of Experience', 0, 35, 3)
        education = st.selectbox('Education Level', C['educations'], index=2)

        st.markdown('<p style="color:#4facfe;font-weight:600;font-size:0.9rem;margin-top:1rem;margin-bottom:0.5rem;">🛠️ SKILLS & CERTS</p>', unsafe_allow_html=True)
        skills = st.multiselect('Skills (select all that apply)', C['skills'], default=['Python', 'SQL'])
        certs = st.multiselect('Certifications (optional)', C['certs'])

        st.markdown('<p style="color:#4facfe;font-weight:600;font-size:0.9rem;margin-top:1rem;margin-bottom:0.5rem;">🌍 LOCATION & COMPANY</p>', unsafe_allow_html=True)
        country = st.selectbox('Country', C['countries'], index=C['countries'].index('United States'))
        city_options = C['countries_cities'].get(country, ['Unknown'])
        city = st.selectbox('City', sorted(city_options))
        industry = st.selectbox('Industry', C['industries'], index=C['industries'].index('Technology'))
        company_size = st.selectbox('Company Size', C['company_sizes'], index=2)

        st.markdown('<p style="color:#4facfe;font-weight:600;font-size:0.9rem;margin-top:1rem;margin-bottom:0.5rem;">📋 EMPLOYMENT</p>', unsafe_allow_html=True)
        employment_type = st.selectbox('Employment Type', C['employment_types'], index=0)
        work_mode = st.radio('Work Mode', C['work_modes'], horizontal=True)

        st.markdown('<br>', unsafe_allow_html=True)
        predict_btn = st.button('🔮 Predict My Salary', use_container_width=True, type='primary')

    return dict(
        job_title=job_title,
        years_experience=years_exp,
        education_level=education,
        skills=skills,
        country=country,
        city=city,
        industry=industry,
        company_size=company_size,
        employment_type=employment_type,
        work_mode=work_mode,
        certifications=certs if certs else ['None'],
        predict=predict_btn,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c5d8e8', family='Inter'),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', showline=False),
)

PRIMARY_COLOR = '#4facfe'
SUCCESS_COLOR = '#43e97b'
WARN_COLOR = '#f6d365'
PURPLE = '#a855f7'
PINK = '#f472b6'

PALETTE = [PRIMARY_COLOR, SUCCESS_COLOR, WARN_COLOR, PURPLE, PINK, '#34d399', '#fb923c']


def cl(**kw):
    """Return CHART_LAYOUT merged with kw, deep-merging xaxis/yaxis to prevent duplicate-key errors."""
    base = dict(CHART_LAYOUT)
    for axis in ('xaxis', 'yaxis'):
        if axis in kw:
            kw[axis] = {**base.pop(axis, {}), **kw[axis]}
    base.update(kw)
    return base


def fmt_currency(val: int) -> str:
    if val >= 1_000_000:
        return f'${val/1_000_000:.2f}M'
    if val >= 1000:
        return f'${val:,.0f}'
    return f'${val}'


# ── Tab 1: Prediction ─────────────────────────────────────────────────────────

def render_prediction_tab(inputs: dict, result: dict):
    if result is None:
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
            <div style="font-size:1.2rem;color:#c5d8e8;font-weight:600;">
                Fill in the details on the left and click<br>
                <span style="color:#4facfe;">"Predict My Salary"</span> to get your estimate.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    sal = result['salary']
    sal_min = result['salary_min']
    sal_max = result['salary_max']
    conf = result['confidence']
    impacts = result['impacts']

    # ── Main salary display ──
    st.markdown(f"""
    <div class="salary-banner">
        <div style="font-size:0.9rem;color:#7a8fa6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
            Estimated Annual Salary
        </div>
        <div class="amount">{fmt_currency(sal)}</div>
        <div class="range">Expected Range: {fmt_currency(sal_min)} - {fmt_currency(sal_max)}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{fmt_currency(sal)}</div>
            <div class="lbl">Annual Salary</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val" style="font-size:1.6rem;">{fmt_currency(sal_min)} - {fmt_currency(sal_max)}</div>
            <div class="lbl">Salary Range</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        c_color = SUCCESS_COLOR if conf >= 80 else WARN_COLOR
        st.markdown(f"""
        <div class="metric-card">
            <div class="val" style="background:linear-gradient(90deg,{c_color},{c_color});">{conf}%</div>
            <div class="lbl">Confidence Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        monthly = sal // 12
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{fmt_currency(monthly)}</div>
            <div class="lbl">Monthly (Est.)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Factor impacts ──
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-title">📊 Salary Factor Impacts</div>', unsafe_allow_html=True)
        factors = {
            '🕐 Experience': impacts.get('experience', 0),
            '🛠️ Skills & Certs': impacts.get('skills', 0),
            '🌍 Location': impacts.get('location', 0),
            '🎓 Education': impacts.get('education', 0),
        }
        max_abs = max(abs(v) for v in factors.values()) or 1
        for label, val in factors.items():
            bar_pct = min(100, int(abs(val) / max_abs * 100))
            direction = '▲' if val >= 0 else '▼'
            color = SUCCESS_COLOR if val >= 0 else '#f87171'
            st.markdown(f"""
            <div class="impact-row">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span class="impact-label">{label}</span>
                    <span style="color:{color};font-weight:700;font-size:0.9rem;">{direction} {abs(val):.1f}%</span>
                </div>
                <div class="impact-bar-bg">
                    <div class="impact-bar-fill" style="width:{bar_pct}%;background:linear-gradient(90deg,{color},{color}88);"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Confidence gauge ──
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">🎯 Prediction Confidence</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=conf,
            number={'suffix': '%', 'font': {'color': '#4facfe', 'size': 32}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#7a8fa6'},
                'bar': {'color': PRIMARY_COLOR, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(248,113,113,0.15)'},
                    {'range': [60, 80], 'color': 'rgba(246,211,101,0.15)'},
                    {'range': [80, 100], 'color': 'rgba(67,233,123,0.15)'},
                ],
                'threshold': {'line': {'color': SUCCESS_COLOR, 'width': 4}, 'value': conf},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c5d8e8'),
            height=220,
            margin=dict(l=20, r=20, t=10, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">📈 5-Year Growth Projection</div>', unsafe_allow_html=True)
        growth_data = result.get('growth_projection', [])
        if growth_data:
            gdf = pd.DataFrame(growth_data)
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=gdf['year'], y=gdf['salary'],
                mode='lines+markers+text',
                text=[fmt_currency(v) for v in gdf['salary']],
                textposition='top center',
                textfont=dict(size=11, color='#c5d8e8'),
                line=dict(color=PRIMARY_COLOR, width=3),
                marker=dict(size=10, color=PRIMARY_COLOR, symbol='circle',
                            line=dict(color='white', width=2)),
                fill='tozeroy',
                fillcolor='rgba(79,172,254,0.08)',
            ))
            fig_growth.update_layout(
                **cl(height=240, showlegend=False, yaxis=dict(tickformat='$,.0f')),
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        # ── Recommended skills ──
        rec = result.get('recommended_skills', [])
        if rec:
            st.markdown('<div class="section-title">💡 High-Impact Skills to Learn</div>', unsafe_allow_html=True)
            chips = ' '.join(f'<span class="skill-chip">✨ {s}</span>' for s in rec[:8])
            st.markdown(chips, unsafe_allow_html=True)

    # ── Export buttons ──
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Export Results</div>', unsafe_allow_html=True)
    ecol1, ecol2, _ = st.columns([1, 1, 3])

    with ecol1:
        csv_data = _build_csv(inputs, result)
        st.download_button(
            '📊 Download CSV',
            data=csv_data,
            file_name='salary_prediction.csv',
            mime='text/csv',
            use_container_width=True,
        )
    with ecol2:
        pdf_data = _build_pdf(inputs, result)
        st.download_button(
            '📄 Download PDF',
            data=pdf_data,
            file_name='salary_prediction.pdf',
            mime='application/pdf',
            use_container_width=True,
        )


def _build_csv(inputs: dict, result: dict) -> bytes:
    rows = [
        ['Field', 'Value'],
        ['Job Title', inputs.get('job_title', '')],
        ['Years of Experience', inputs.get('years_experience', '')],
        ['Education Level', inputs.get('education_level', '')],
        ['Skills', ', '.join(inputs.get('skills', []))],
        ['Country', inputs.get('country', '')],
        ['City', inputs.get('city', '')],
        ['Industry', inputs.get('industry', '')],
        ['Company Size', inputs.get('company_size', '')],
        ['Employment Type', inputs.get('employment_type', '')],
        ['Work Mode', inputs.get('work_mode', '')],
        ['Certifications', ', '.join(inputs.get('certifications', []))],
        [],
        ['Prediction', ''],
        ['Estimated Annual Salary (USD)', result['salary']],
        ['Salary Range Min (USD)', result['salary_min']],
        ['Salary Range Max (USD)', result['salary_max']],
        ['Confidence Score (%)', result['confidence']],
        ['Experience Impact (%)', result['impacts']['experience']],
        ['Skills Impact (%)', result['impacts']['skills']],
        ['Location Impact (%)', result['impacts']['location']],
        ['Education Impact (%)', result['impacts']['education']],
    ]
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, header=False).encode()


def _build_pdf(inputs: dict, result: dict) -> bytes:
    try:
        from fpdf import FPDF

        def _safe(s):
            return (str(s)
                .replace('–', '-').replace('—', '-')
                .replace('‘', "'").replace('’', "'")
                .replace('“', '"').replace('”', '"')
                .replace('…', '...'))

        class PDF(FPDF):
            def header(self):
                self.set_fill_color(15, 32, 39)
                self.rect(0, 0, 210, 30, 'F')
                self.set_text_color(79, 172, 254)
                self.set_font('Helvetica', 'B', 18)
                self.cell(0, 15, 'AI Job Salary Estimator - Prediction Report', ln=True, align='C')
                self.set_text_color(168, 192, 204)
                self.set_font('Helvetica', '', 9)
                self.cell(0, 8, f'Generated on {datetime.datetime.now().strftime("%B %d, %Y")}', ln=True, align='C')
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_text_color(100, 120, 140)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, f'AI-Powered Salary Estimator  |  Page {self.page_no()}', align='C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def section(title):
            pdf.set_fill_color(26, 42, 61)
            pdf.set_text_color(79, 172, 254)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 9, _safe(title), ln=True, fill=True)
            pdf.ln(2)

        def row(label, value, highlight=False):
            pdf.set_font('Helvetica', '', 10)
            if highlight:
                pdf.set_text_color(67, 233, 123)
                pdf.set_font('Helvetica', 'B', 12)
            else:
                pdf.set_text_color(200, 215, 230)
            pdf.cell(80, 8, _safe(label), border='B')
            pdf.cell(0, 8, _safe(value), border='B', ln=True)
            pdf.set_text_color(200, 215, 230)
            pdf.set_font('Helvetica', '', 10)

        section('Role Details')
        row('Job Title', inputs.get('job_title', ''))
        row('Experience', f"{inputs.get('years_experience', 0)} years")
        row('Education', inputs.get('education_level', ''))
        row('Skills', ', '.join(inputs.get('skills', [])) or 'None')
        row('Certifications', ', '.join(inputs.get('certifications', ['None'])))
        pdf.ln(4)

        section('Location & Company')
        row('Country', inputs.get('country', ''))
        row('City', inputs.get('city', ''))
        row('Industry', inputs.get('industry', ''))
        row('Company Size', inputs.get('company_size', ''))
        row('Employment Type', inputs.get('employment_type', ''))
        row('Work Mode', inputs.get('work_mode', ''))
        pdf.ln(4)

        section('Salary Prediction')
        row('Estimated Annual Salary', f"USD {result['salary']:,}", highlight=True)
        row('Salary Range', f"USD {result['salary_min']:,} - {result['salary_max']:,}")
        row('Confidence Score', f"{result['confidence']}%")
        row('Monthly Estimate', f"USD {result['salary'] // 12:,}")
        pdf.ln(4)

        section('Factor Impacts')
        for k, v in result['impacts'].items():
            direction = '+' if v >= 0 else ''
            row(k.capitalize() + ' Impact', f'{direction}{v:.1f}%')
        pdf.ln(4)

        section('Recommended Skills to Learn')
        skills_text = ', '.join(result.get('recommended_skills', [])[:8])
        pdf.set_text_color(200, 215, 230)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 8, _safe(skills_text or 'N/A'))

        return bytes(pdf.output())
    except Exception:
        return b'%PDF-1.4\n% PDF export unavailable'


# ── Tab 2: Analytics ──────────────────────────────────────────────────────────

def render_analytics_tab(df: pd.DataFrame):
    if df is None:
        st.info('Dataset not available.')
        return

    st.markdown('<div class="section-title">📊 Market Salary Analytics</div>', unsafe_allow_html=True)

    # ── Row 1: Distribution + Experience vs Salary ──
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df, x='annual_salary_usd', nbins=60,
            title='Salary Distribution',
            color_discrete_sequence=[PRIMARY_COLOR],
            labels={'annual_salary_usd': 'Annual Salary (USD)'},
        )
        fig.update_layout(**CHART_LAYOUT, height=320)
        fig.update_traces(opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        exp_df = df.groupby('years_experience')['annual_salary_usd'].median().reset_index()
        exp_df.columns = ['Years', 'Median Salary']
        fig2 = px.line(
            exp_df, x='Years', y='Median Salary',
            title='Experience vs Median Salary',
            color_discrete_sequence=[SUCCESS_COLOR],
            markers=True,
        )
        fig2.update_layout(**CHART_LAYOUT, height=320)
        fig2.update_traces(line_width=3, marker_size=6)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Education + Industry ──
    c3, c4 = st.columns(2)

    with c3:
        edu_order = [
            'High School / GED', "Associate's Degree", "Bachelor's Degree",
            "Master's Degree", 'PhD / Doctorate', 'Bootcamp Graduate',
        ]
        edu_df = df.groupby('education_level')['annual_salary_usd'].median().reindex(edu_order).dropna().reset_index()
        edu_df.columns = ['Education', 'Median Salary']
        fig3 = px.bar(
            edu_df, x='Education', y='Median Salary',
            title='Education Level vs Median Salary',
            color='Median Salary',
            color_continuous_scale=[[0, '#203a43'], [0.5, PRIMARY_COLOR], [1, SUCCESS_COLOR]],
        )
        fig3.update_layout(**cl(height=320, coloraxis_showscale=False, xaxis=dict(tickangle=-20)))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        ind_df = df.groupby('industry')['annual_salary_usd'].median().sort_values(ascending=True).reset_index()
        ind_df.columns = ['Industry', 'Median Salary']
        fig4 = px.bar(
            ind_df, x='Median Salary', y='Industry',
            orientation='h',
            title='Industry-wise Median Salary',
            color='Median Salary',
            color_continuous_scale=[[0, '#203a43'], [1, PURPLE]],
        )
        fig4.update_layout(**CHART_LAYOUT, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Country + Company size ──
    c5, c6 = st.columns(2)

    with c5:
        cnt_df = (
            df.groupby('country')['annual_salary_usd']
            .median()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        cnt_df.columns = ['Country', 'Median Salary']
        fig5 = px.bar(
            cnt_df, x='Median Salary', y='Country',
            orientation='h',
            title='Top Countries by Median Salary',
            color='Median Salary',
            color_continuous_scale=[[0, '#203a43'], [1, WARN_COLOR]],
        )
        fig5.update_layout(**CHART_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        cs_order = [
            'Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)',
            'Large (1001-5000)', 'Enterprise (5000+)',
        ]
        cs_df = df.groupby('company_size')['annual_salary_usd'].median().reindex(cs_order).dropna().reset_index()
        cs_df.columns = ['Size', 'Median Salary']
        fig6 = px.bar(
            cs_df, x='Size', y='Median Salary',
            title='Company Size vs Median Salary',
            color='Median Salary',
            color_continuous_scale=[[0, '#203a43'], [1, PINK]],
        )
        fig6.update_layout(**cl(height=320, coloraxis_showscale=False, xaxis=dict(tickangle=-15)))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Work mode + Skill frequency ──
    c7, c8 = st.columns(2)

    with c7:
        wm_df = df.groupby('work_mode')['annual_salary_usd'].median().reset_index()
        wm_df.columns = ['Work Mode', 'Median Salary']
        fig7 = px.pie(
            wm_df, names='Work Mode', values='Median Salary',
            title='Salary by Work Mode',
            color_discrete_sequence=PALETTE,
            hole=0.5,
        )
        fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c5d8e8'),
                           height=300, margin=dict(l=10, r=10, t=40, b=10))
        fig7.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        skill_counts = {}
        for row in df['skills'].dropna():
            for s in str(row).split(';'):
                s = s.strip()
                if s:
                    skill_counts[s] = skill_counts.get(s, 0) + 1
        top_sk = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        sk_df = pd.DataFrame(top_sk, columns=['Skill', 'Count'])
        fig8 = px.bar(
            sk_df.sort_values('Count'), x='Count', y='Skill',
            orientation='h',
            title='Most In-Demand Skills in Dataset',
            color='Count',
            color_continuous_scale=[[0, '#203a43'], [1, SUCCESS_COLOR]],
        )
        fig8.update_layout(**CHART_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig8, use_container_width=True)


# ── Tab 3: Insights ───────────────────────────────────────────────────────────

def render_insights_tab(df: pd.DataFrame, inputs: dict, result: dict):
    if df is None:
        st.info('Dataset not available.')
        return

    st.markdown('<div class="section-title">💡 Career & Market Insights</div>', unsafe_allow_html=True)

    # ── Top 10 paying job titles ──
    c1, c2 = st.columns(2)
    with c1:
        top_jobs = (
            df.groupby('job_title')['annual_salary_usd']
            .median()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_jobs.columns = ['Job Title', 'Median Salary']
        fig = px.bar(
            top_jobs.sort_values('Median Salary'),
            x='Median Salary', y='Job Title',
            orientation='h',
            title='Top 10 Highest-Paying Job Titles',
            color='Median Salary',
            color_continuous_scale=[[0, '#203a43'], [1, WARN_COLOR]],
        )
        fig.update_layout(**CHART_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        from generate_data import SKILL_PREMIUMS
        skill_items = sorted(SKILL_PREMIUMS.items(), key=lambda x: x[1], reverse=True)[:12]
        sp_df = pd.DataFrame(skill_items, columns=['Skill', 'Premium'])
        sp_df['Premium %'] = (sp_df['Premium'] * 100).round(1)
        fig2 = px.bar(
            sp_df.sort_values('Premium %'),
            x='Premium %', y='Skill',
            orientation='h',
            title='Top 10 Salary-Boosting Skills',
            color='Premium %',
            color_continuous_scale=[[0, '#203a43'], [1, PRIMARY_COLOR]],
        )
        fig2.update_layout(**CHART_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Salary by experience band ──
    st.markdown('<div class="section-title">📈 Salary Growth by Experience Band</div>', unsafe_allow_html=True)
    bins = [0, 2, 5, 8, 12, 18, 35]
    labels = ['0-2 yrs', '3-5 yrs', '6-8 yrs', '9-12 yrs', '13-18 yrs', '18+ yrs']
    df2 = df.copy()
    df2['exp_band'] = pd.cut(df2['years_experience'], bins=bins, labels=labels)
    band_df = (
        df2.groupby('exp_band', observed=True)['annual_salary_usd']
        .agg(['median', 'mean', 'std'])
        .reset_index()
    )
    band_df.columns = ['Band', 'Median', 'Mean', 'Std']

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=band_df['Band'], y=band_df['Median'],
                          name='Median', marker_color=PRIMARY_COLOR))
    fig3.add_trace(go.Bar(x=band_df['Band'], y=band_df['Mean'],
                          name='Mean', marker_color=WARN_COLOR, opacity=0.75))
    fig3.update_layout(**CHART_LAYOUT, height=320, barmode='group',
                       legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Career path ──
    st.markdown('<div class="section-title">🗺️ Suggested Career Paths</div>', unsafe_allow_html=True)
    from predict import get_career_paths
    paths = get_career_paths(inputs.get('job_title', ''))
    for i, path in enumerate(paths):
        color = [PRIMARY_COLOR, SUCCESS_COLOR, WARN_COLOR][i % 3]
        steps = path.split(' → ')
        chips = ' &rarr; '.join(
            f'<span style="background:rgba(79,172,254,0.12);border:1px solid {color}44;'
            f'border-radius:20px;padding:0.2rem 0.8rem;color:{color};font-weight:600;font-size:0.88rem;">'
            f'{s}</span>'
            for s in steps
        )
        st.markdown(
            f'<div class="info-card" style="margin:0.4rem 0;">{chips}</div>',
            unsafe_allow_html=True
        )

    # ── Salary by industry x experience ──
    st.markdown('<div class="section-title">🏭 Industry Salary Heatmap (Experience Bands)</div>', unsafe_allow_html=True)
    pivot = df2.pivot_table(
        values='annual_salary_usd', index='industry',
        columns='exp_band', aggfunc='median', observed=True
    ).dropna(how='all').fillna(0)
    fig4 = px.imshow(
        pivot,
        color_continuous_scale='Blues',
        title='Median Salary: Industry × Experience Band',
        aspect='auto',
        labels=dict(color='Median Salary'),
    )
    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#c5d8e8'), height=420,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)


# ── Tab 4: Compare Roles ──────────────────────────────────────────────────────

def render_compare_tab(C: dict):
    st.markdown('<div class="section-title">🔄 Compare Multiple Job Roles</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#7a8fa6;font-size:0.9rem;">Configure up to 3 scenarios and compare estimated salaries side-by-side.</p>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    scenarios = []
    colors = [PRIMARY_COLOR, SUCCESS_COLOR, WARN_COLOR]

    for i, col in enumerate(cols):
        with col:
            color = colors[i]
            st.markdown(
                f'<div style="border:2px solid {color}44;border-radius:12px;padding:1rem;margin-bottom:0.5rem;">',
                unsafe_allow_html=True
            )
            st.markdown(f'<b style="color:{color};">Scenario {i+1}</b>', unsafe_allow_html=True)
            jt = st.selectbox('Job Title', C['job_titles'],
                              index=min(i * 5, len(C['job_titles']) - 1),
                              key=f'cmp_jt_{i}')
            ye = st.slider('Years Exp', 0, 35, [2, 5, 10][i], key=f'cmp_ye_{i}')
            edu = st.selectbox('Education', C['educations'],
                               index=2, key=f'cmp_edu_{i}')
            cty = st.selectbox('Country', C['countries'],
                               index=C['countries'].index('United States'), key=f'cmp_cty_{i}')
            ind = st.selectbox('Industry', C['industries'],
                               index=C['industries'].index('Technology'), key=f'cmp_ind_{i}')
            wm = st.radio('Work Mode', C['work_modes'], key=f'cmp_wm_{i}', horizontal=True)
            sk = st.multiselect('Skills', C['skills'], default=['Python', 'SQL'],
                                key=f'cmp_sk_{i}')
            scenarios.append(dict(
                label=f'Scenario {i+1}: {jt}',
                job_title=jt, years_experience=ye, education_level=edu,
                country=cty, city=C['countries_cities'][cty][0],
                industry=ind, company_size='Medium (201-1000)',
                employment_type='Full-time', work_mode=wm,
                skills=sk, certifications=['None'],
                color=color,
            ))
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button('⚡ Compare Now', type='primary', use_container_width=True):
        from predict import predict_salary
        results_cmp = []
        for s in scenarios:
            try:
                r = predict_salary(s)
                results_cmp.append({'label': s['label'], 'color': s['color'], **r})
            except Exception as e:
                st.warning(f"Error predicting {s['label']}: {e}")

        if results_cmp:
            st.markdown('<br>', unsafe_allow_html=True)

            # ── Bar chart comparison ──
            fig = go.Figure()
            for r in results_cmp:
                fig.add_trace(go.Bar(
                    name=r['label'],
                    x=[r['label']],
                    y=[r['salary']],
                    marker_color=r['color'],
                    text=[fmt_currency(r['salary'])],
                    textposition='outside',
                    error_y=dict(
                        type='data',
                        array=[r['salary_max'] - r['salary']],
                        arrayminus=[r['salary'] - r['salary_min']],
                        visible=True,
                        color='rgba(255,255,255,0.5)',
                        thickness=2,
                        width=8,
                    ),
                ))
            fig.update_layout(
                **cl(title='Estimated Annual Salary Comparison', height=380,
                     showlegend=False, yaxis=dict(tickformat='$,.0f')),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Radar chart ──
            cats = ['Experience Impact', 'Skills Impact', 'Confidence', 'Salary (normalized)']
            max_sal = max(r['salary'] for r in results_cmp) or 1
            fig_r = go.Figure()
            for r in results_cmp:
                vals = [
                    min(100, abs(r['impacts']['experience'])) * 3,
                    min(100, r['impacts']['skills']) * 3,
                    r['confidence'],
                    r['salary'] / max_sal * 100,
                ]
                vals += [vals[0]]
                cats_loop = cats + [cats[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats_loop,
                    fill='toself',
                    name=r['label'],
                    line_color=r['color'],
                    fillcolor=r['color'].replace(')', ',0.12)').replace('rgb', 'rgba') if 'rgb' in r['color'] else r['color'] + '22',
                    opacity=0.85,
                ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c5d8e8'),
                height=380,
                title='Scenario Comparison Radar',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # ── Comparison table ──
            st.markdown('<div class="section-title">📋 Detailed Comparison Table</div>', unsafe_allow_html=True)
            tbl = []
            for r in results_cmp:
                tbl.append({
                    'Scenario': r['label'],
                    'Annual Salary': fmt_currency(r['salary']),
                    'Range': f"{fmt_currency(r['salary_min'])} - {fmt_currency(r['salary_max'])}",
                    'Confidence': f"{r['confidence']}%",
                    'Exp Impact': f"{r['impacts']['experience']:+.1f}%",
                    'Skills Impact': f"{r['impacts']['skills']:+.1f}%",
                })
            st.dataframe(
                pd.DataFrame(tbl).set_index('Scenario'),
                use_container_width=True,
            )


# ── Model Results ─────────────────────────────────────────────────────────────

def render_model_tab(training_results: dict):
    st.markdown('<div class="section-title">🤖 ML Model Training Results</div>', unsafe_allow_html=True)
    if not training_results:
        st.info('Model not trained yet.')
        return

    results = training_results.get('results', [])
    best = training_results.get('best', {})

    st.markdown(f"""
    <div class="info-card">
        ✅  <b style="color:{SUCCESS_COLOR};">Best Model: {best.get('name', 'N/A')}</b>
        &nbsp;|&nbsp; R² = {best.get('r2', 0):.4f}
        &nbsp;|&nbsp; MAE = ${best.get('mae', 0):,.0f}
        &nbsp;|&nbsp; RMSE = ${best.get('rmse', 0):,.0f}
    </div>
    """, unsafe_allow_html=True)

    df_r = pd.DataFrame(results)
    df_r.columns = ['Model', 'R²', 'MAE (USD)', 'RMSE (USD)', 'CV R²', 'Train Time (s)']

    # ── R² comparison bar ──
    fig = go.Figure()
    colors_bar = [SUCCESS_COLOR if r['name'] == best.get('name') else PRIMARY_COLOR for r in results]
    fig.add_trace(go.Bar(
        x=df_r['Model'], y=df_r['R²'],
        marker_color=colors_bar,
        text=[f'{v:.4f}' for v in df_r['R²']],
        textposition='outside',
    ))
    fig.update_layout(**cl(title='R² Score Comparison', height=320,
                           yaxis=dict(range=[0, 1.05]), showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_r['Model'], y=df_r['MAE (USD)'],
                              marker_color=WARN_COLOR,
                              text=[f'${v:,.0f}' for v in df_r['MAE (USD)']],
                              textposition='outside'))
        fig2.update_layout(**cl(title='Mean Absolute Error', height=300,
                               yaxis=dict(tickformat='$,.0f')))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df_r['Model'], y=df_r['RMSE (USD)'],
                              marker_color=PURPLE,
                              text=[f'${v:,.0f}' for v in df_r['RMSE (USD)']],
                              textposition='outside'))
        fig3.update_layout(**cl(title='Root Mean Squared Error', height=300,
                               yaxis=dict(tickformat='$,.0f')))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">📋 Full Results Table</div>', unsafe_allow_html=True)
    st.dataframe(
        df_r.set_index('Model').style.highlight_max(subset=['R²', 'CV R²'], color='#1a3a2a')
                                     .highlight_min(subset=['MAE (USD)', 'RMSE (USD)'], color='#1a3a2a')
                                     .format({'R²': '{:.4f}', 'CV R²': '{:.4f}',
                                              'MAE (USD)': '${:,.0f}', 'RMSE (USD)': '${:,.0f}'}),
        use_container_width=True,
    )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>💼 AI Job Salary Estimator</h1>
        <p>Machine-learning powered predictions · No sign-up required · Instant results</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-train on cold start (Streamlit Cloud: model is not in git) ──
    if not os.path.exists('models/best_model.pkl'):
        run_initialization()
        st.stop()

    C = get_constants()
    inputs = render_sidebar(C)

    # ── Run prediction ──
    result = None
    if inputs['predict']:
        with st.spinner('Computing salary estimate…'):
            try:
                from predict import predict_salary
                result = predict_salary(inputs)
                st.session_state['last_result'] = result
                st.session_state['last_inputs'] = inputs
            except Exception as e:
                st.error(f'Prediction error: {e}')

    if result is None and 'last_result' in st.session_state:
        result = st.session_state['last_result']
        inputs_for_display = st.session_state.get('last_inputs', inputs)
    else:
        inputs_for_display = inputs

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '🎯 Predict',
        '📊 Analytics',
        '💡 Insights',
        '🔄 Compare',
        '🤖 Model Results',
    ])

    df = load_dataset()

    with tab1:
        render_prediction_tab(inputs_for_display, result)

    with tab2:
        render_analytics_tab(df)

    with tab3:
        render_insights_tab(df, inputs_for_display, result)

    with tab4:
        render_compare_tab(C)

    with tab5:
        render_model_tab(get_training_results())

    # Footer
    st.markdown("""
    <div class="footer">
        AI Job Salary Estimator · Built with Streamlit, scikit-learn, XGBoost, LightGBM & CatBoost<br>
        Predictions are statistical estimates based on synthetic training data — not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
