from pathlib import Path
import html

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="StudySmart AI | Student Performance Research",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stMain"], .main { background: #ffffff !important; }
    [data-testid="stHeader"] { background: rgba(255,255,255,.96) !important; }
    [data-testid="stSidebar"] { background: #f8faff !important; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
    html, body, [class*="st-"] { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
    h1, h2, h3, h4, p, label { color: #152541; }
    h1 { font-weight: 760 !important; letter-spacing: -0.045em !important; }
    h2, h3 { letter-spacing: -0.025em !important; }
    [data-testid="stCaptionContainer"] { color: #64748b !important; }
    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 2.4rem; }
    .brand-icon { background: #edf3ff; color: #245ac2; width: 44px; height: 44px;
        border-radius: 13px; display: flex; align-items: center; justify-content: center;
        font-size: 23px; }
    .brand-name { color: #152541; font-size: 19px; font-weight: 760; line-height: 1.1; }
    .brand-sub { color: #7a879a; font-size: 12px; margin-top: 4px; }
    .eyebrow { color: #3269c7; text-transform: uppercase; letter-spacing: .13em;
        font-size: 12px; font-weight: 750; margin-bottom: 12px; }
    .hero-title { color: #152541; font-size: clamp(34px, 4.5vw, 55px); font-weight: 800;
        line-height: 1.11; letter-spacing: -.052em; margin: 0 0 16px; }
    .hero-copy { color: #64748b; font-size: 17px; max-width: 720px; line-height: 1.65; margin-bottom: 28px; }
    .stat-card { background: #f7f9fd; border: 1px solid #e9eef6; border-radius: 16px;
        padding: 22px 24px; min-height: 110px; margin-bottom: 12px; }
    .stat-label { color: #64748b; font-size: 13px; margin-bottom: 9px; }
    .stat-value { color: #152541; font-size: 29px; font-weight: 760; letter-spacing: -.035em; }
    .section-kicker { color: #64748b; font-size: 14px; margin-bottom: 22px; }
    .result-panel { border: 1px solid #dce6f8; background: #f5f8ff;
        border-radius: 18px; padding: 26px 30px; margin-top: 20px; }
    .result-label { color: #526580; font-size: 14px; font-weight: 650; }
    .result-number { color: #215bc4; font-size: 66px; line-height: 1.2;
        font-weight: 800; letter-spacing: -.065em; }
    .result-number span { color: #7184a1; font-size: 24px; font-weight: 500; letter-spacing: 0; }
    .result-note { color: #53647c; font-size: 13px; line-height: 1.7; }
    .soft-note { background: #f8faff; border: 1px solid #e8edf6; border-radius: 12px;
        padding: 15px 18px; color: #54647a; font-size: 13px; line-height: 1.65; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: #245ec9 !important; color: white !important; border: 1px solid #245ec9 !important;
        border-radius: 10px !important; padding: .72rem 1.4rem !important; font-weight: 650 !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #174cae !important; border-color: #174cae !important;
    }
    div[data-testid="stMetric"] { background: #f8faff; border: 1px solid #e9eef6;
        padding: 18px 20px; border-radius: 14px; }
    div[data-testid="stMetricValue"] { color: #152541; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    div[data-testid="stImage"] img { border-radius: 12px; }
    hr { border-color: #e8edf5 !important; }
    @media (max-width: 700px) { .block-container { padding-top: 1.2rem; }
        .result-number { font-size: 52px; } .hero-copy { font-size: 15px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
    <div class="brand-icon">🎓</div>
    <div><div class="brand-name">StudySmart AI</div>
    <div class="brand-sub">Independent machine learning research · 2026</div></div>
</div>
<div class="eyebrow">Explore the data. Understand the predictions.</div>
<div class="hero-title">A clearer picture of<br>academic performance.</div>
<div class="hero-copy">Explore how student characteristics and previous grades relate to final exam results, using models trained on real educational data.</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

habits_path = RESULTS / "habits_model.joblib"
history_path = RESULTS / "history_model.joblib"
if not habits_path.is_file() or not history_path.is_file():
    st.error("Trained models are missing from the results/ folder.")
    st.code("python train.py --data data/student-por.csv", language="bash")
    st.stop()

try:
    habits_bundle = load_model(str(habits_path))
    history_bundle = load_model(str(history_path))
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}")
    st.stop()

metrics_h = habits_bundle.get("metrics", {})
metrics_p = history_bundle.get("metrics", {})

stats = st.columns(3)
stat_data = [
    ("Student records", "649"),
    ("Prediction models", "2"),
    ("Best test MAE", f"{metrics_p.get('test_mae', 0.75):.2f} / 20"),
]
for column, (label, value) in zip(stats, stat_data):
    with column:
        st.markdown(f'<div class="stat-card"><div class="stat-label">{html.escape(label)}</div><div class="stat-value">{html.escape(value)}</div></div>', unsafe_allow_html=True)

st.write("")
predict_tab, analysis_tab, about_tab = st.tabs(["✦  Grade prediction", "▥  Model analysis", "ⓘ  About the research"])

with predict_tab:
    st.write("")
    st.subheader("Make a prediction")
    st.markdown('<div class="section-kicker">Choose a model and enter a student profile. The model will estimate a final grade on a 0–20 scale.</div>', unsafe_allow_html=True)
    scenario = st.radio(
        "Prediction approach",
        ["Habits only", "Habits + previous grades"],
        horizontal=True,
        help="The second model also uses first- and second-period grades (G1 and G2).",
    )
    history_mode = scenario == "Habits + previous grades"
    bundle = history_bundle if history_mode else habits_bundle
    feature_columns = bundle["columns"]

    with st.form("prediction_form", border=True):
        st.markdown("#### Study habits and wellbeing")
        left, right = st.columns(2, gap="large")
        with left:
            studytime = st.select_slider(
                "Weekly study time", options=[1, 2, 3, 4], value=2,
                format_func=lambda v: {1: "Under 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "Over 10 hours"}[v],
                help="These are the original study-time categories in the UCI dataset.",
            )
            absences = st.slider("School absences", 0, 50, 4)
            freetime = st.slider("Free time after school", 1, 5, 3, help="1 = very little, 5 = very much")
        with right:
            goout = st.slider("Going out with friends", 1, 5, 3, help="1 = very little, 5 = very much")
            health = st.slider("Self-rated health", 1, 5, 3, help="1 = very bad, 5 = very good")

        st.markdown("#### School and home environment")
        left2, right2 = st.columns(2, gap="large")
        with left2:
            activities = st.selectbox("Extracurricular activities", ["yes", "no"], format_func=lambda x: x.title())
            schoolsup = st.selectbox("Extra school support", ["no", "yes"], format_func=lambda x: x.title())
            famsup = st.selectbox("Family educational support", ["yes", "no"], format_func=lambda x: x.title())
        with right2:
            internet = st.selectbox("Internet access at home", ["yes", "no"], format_func=lambda x: x.title())
            higher = st.selectbox("Plans for higher education", ["yes", "no"], format_func=lambda x: x.title())

        if history_mode:
            st.markdown("#### Previous academic performance")
            st.caption("These are grades from earlier periods of the same course, not separate subjects.")
            grade_left, grade_right = st.columns(2, gap="large")
            with grade_left:
                G1 = st.slider("First-period grade (G1)", 0, 20, 12)
            with grade_right:
                G2 = st.slider("Second-period grade (G2)", 0, 20, 12)

        submitted = st.form_submit_button("Predict final grade  →", type="primary", use_container_width=True)

    if submitted:
        entered = {
            "studytime": studytime, "absences": absences, "freetime": freetime,
            "goout": goout, "health": health, "activities": activities,
            "schoolsup": schoolsup, "famsup": famsup, "internet": internet, "higher": higher,
        }
        if history_mode:
            entered.update({"G1": G1, "G2": G2})
        missing = [c for c in feature_columns if c not in entered]
        if missing:
            st.error(f"The trained model expects additional inputs: {', '.join(missing)}")
        else:
            try:
                features = pd.DataFrame([{c: entered[c] for c in feature_columns}])
                prediction = float(bundle["pipeline"].predict(features)[0])
                prediction = max(0.0, min(20.0, prediction))
                st.session_state["prediction_result"] = {
                    "value": prediction,
                    "scenario": scenario,
                    "mae": bundle.get("metrics", {}).get("test_mae"),
                    "algorithm": str(bundle.get("metrics", {}).get("selected_model", "Trained regression model")),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("prediction_result")
    if result is not None:
        if result["scenario"] == scenario:
            st.markdown(
                f'<div class="result-panel"><div class="result-label">Estimated final grade</div>'
                f'<div class="result-number">{result["value"]:.1f} <span>/ 20</span></div>'
                f'<div class="result-note">Model: {html.escape(result["scenario"])}'
                f' · Algorithm: {html.escape(result["algorithm"])}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(result["value"] / 20)
            if result["mae"] is not None:
                st.caption(f"On the held-out test data, this model's average absolute error was {result['mae']:.2f} grade points. This is not an individual prediction guarantee.")
        else:
            st.caption("You switched models. Click 'Predict final grade' to calculate a new result.")
    else:
        st.caption("Complete the form and click 'Predict final grade' to see the result.")

    st.write("")
    st.markdown('<div class="soft-note"><strong>Research disclaimer.</strong> This tool illustrates statistical patterns in historical student data. It is not an individual academic assessment, cannot establish cause and effect, and does not measure sleep or recreational screen time.</div>', unsafe_allow_html=True)

with analysis_tab:
    st.write("")
    st.subheader("How well do the models perform?")
    st.caption("Both models were evaluated on 130 held-out student records. Lower MAE is better; higher R² is better.")
    comparison = pd.DataFrame([
        {"Model": "Habits only", "Test MAE (grade points)": metrics_h.get("test_mae"), "Test R²": metrics_h.get("test_r2")},
        {"Model": "Habits + previous grades", "Test MAE (grade points)": metrics_p.get("test_mae"), "Test R²": metrics_p.get("test_r2")},
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.markdown("#### Explore the evaluation graphs")
    graph_model = st.segmented_control(
        "View graphs for", ["Habits only", "Habits + previous grades"],
        default="Habits + previous grades", key="analysis_model",
    )
    prefix = "history" if graph_model == "Habits + previous grades" else "habits"
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("**Predicted vs. actual grades**")
        chart = RESULTS / f"{prefix}_predictions.png"
        if chart.is_file():
            st.image(str(chart), use_container_width=True)
        else:
            st.warning(f"Graph not found: results/{chart.name}")
        st.caption("Each point represents a student. Points closer to the ideal diagonal indicate more accurate predictions.")
    with chart_right:
        st.markdown("**Permutation feature importance**")
        chart = RESULTS / f"{prefix}_importance.png"
        if chart.is_file():
            st.image(str(chart), use_container_width=True)
        else:
            st.warning(f"Graph not found: results/{chart.name}")
        st.caption("A larger increase in MAE after shuffling a feature means the model relied more on that feature.")
    st.info("Previous grades are strongly related to the final grade because they measure performance in the same course. Feature importance is not evidence of causation.")

with about_tab:
    st.write("")
    st.subheader("About StudySmart AI")
    st.write("StudySmart AI is an educational machine learning research project exploring how student characteristics and earlier academic performance relate to final grades.")
    st.markdown("#### The dataset")
    st.write("The project uses the UCI Student Performance dataset: 649 records from a Portuguese-language course in Portuguese secondary schools. The final target grade (G3) ranges from 0 to 20.")
    st.markdown("#### The two experiments")
    st.write("**Habits only:** predicts final grades from recorded student habits and educational circumstances. **Habits + previous grades:** adds G1 and G2, two earlier grades from the same course.")
    st.markdown("#### How to interpret the results")
    st.write("MAE measures the average absolute prediction error in grade points. R² measures how much the model reduces squared prediction error relative to predicting the average grade on the test set.")
    st.markdown("#### Limitations")
    st.write("The data come from a limited historical sample and may not generalise to other schools or countries. The dataset does not record sleep duration or recreational screen time. Observed relationships are predictive, not necessarily causal.")
    st.markdown("[View the dataset on UCI](https://archive.ics.uci.edu/dataset/320/student+performance)  ·  [Explore the project on GitHub](https://github.com/omarsns1/StudySmart-AI)")

st.divider()
st.caption("StudySmart AI · Independent ML research project · Educational demonstration only")
