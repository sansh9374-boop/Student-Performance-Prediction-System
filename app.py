"""Streamlit dashboard for Student Performance Prediction System."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import load_data, FEATURES
from model import load_models
from recommend import generate_recommendations

DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

@st.cache_resource
def get_models():
    return load_models(MODEL_DIR)

df = get_data()
classifier, regressor = get_models()

st.title("🎓 Student Performance Prediction System")
st.caption("Machine learning dashboard for academic analytics, prediction and recommendations.")

with st.sidebar:
    st.header("Student Inputs")
    gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
    study_mode = st.selectbox("Study mode", sorted(df["study_mode"].dropna().unique()))
    subject = st.selectbox("Subject", sorted(df["subject"].dropna().unique()))
    attendance = st.slider("Attendance (%)", 40, 100, 80)
    study_hours = st.slider("Study hours/day", 0.5, 9.0, 4.0, 0.5)
    internal_marks = st.slider("Internal marks", 20.0, 100.0, 65.0, 1.0)
    participation = st.slider("Participation (1-10)", 1.0, 10.0, 7.0, 0.5)
    previous_score = st.slider("Previous score", 20.0, 100.0, 70.0, 1.0)
    assignment_completion = st.slider("Assignment completion (%)", 20, 100, 80)
    weak_subjects = st.slider("Number of weak subjects", 0, 5, 1)

input_row = pd.DataFrame([{
    "attendance": attendance,
    "study_hours": study_hours,
    "internal_marks": internal_marks,
    "participation": participation,
    "previous_score": previous_score,
    "assignment_completion": assignment_completion,
    "weak_subjects": weak_subjects,
    "gender": gender,
    "study_mode": study_mode,
    "subject": subject
}])

pred_level = classifier.predict(input_row[FEATURES])[0]
pred_score = float(regressor.predict(input_row[FEATURES])[0])
recs = generate_recommendations(input_row.iloc[0].to_dict())

c1, c2, c3 = st.columns(3)
c1.metric("Predicted Score", f"{pred_score:.1f}/100")
c2.metric("Performance Level", pred_level)
c3.metric("Attendance", f"{attendance:.0f}%")

st.subheader("Performance Insights")
col1, col2 = st.columns(2)

with col1:
    st.write("**Student profile**")
    st.dataframe(input_row.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.write("**Recommendations**")
    for rec in recs:
        st.info(rec)

st.subheader("Dataset Analytics")
level_counts = df["performance_level"].value_counts().reindex(["High", "Average", "Low"]).fillna(0)
st.bar_chart(level_counts)

st.subheader("Feature Relationships")
chart_df = df[["attendance", "study_hours", "internal_marks", "participation", "previous_score", "final_score"]].corr()["final_score"].drop("final_score")
st.bar_chart(chart_df.sort_values(ascending=False))

st.subheader("Student Comparison")
comparison = df.groupby("performance_level")["final_score"].agg(["count", "mean", "min", "max"]).reindex(["High", "Average", "Low"])
st.dataframe(comparison.round(2), use_container_width=True)
