import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model1.pkl")

model = load_model()

# ── Title & Description ────────────────────────────────────────────────────────
st.title("💰 Adult Income Predictor")
st.markdown(
    "Predict whether a person's annual income is **>50K** or **≤50K** "
    "based on demographic and employment information."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("📋 Enter Person Details")

# ── Input Fields ───────────────────────────────────────────────────────────────
age = st.sidebar.slider("Age", 18, 90, 35)

workclass = st.sidebar.selectbox("Work Class", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov",
    "Without-pay", "Never-worked"
])

fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=189778, step=1000)

education = st.sidebar.selectbox("Education", [
    "Bachelors", "Some-college", "11th", "HS-grad",
    "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
    "7th-8th", "12th", "Masters", "1st-4th",
    "10th", "Doctorate", "5th-6th", "Preschool"
])

educational_num = st.sidebar.slider("Education Years (Numeric)", 1, 16, 10)

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent",
    "Married-AF-spouse"
])

occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service",
    "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband",
    "Not-in-family", "Other-relative", "Unmarried"
])

race = st.sidebar.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
capital_loss  = st.sidebar.number_input("Capital Loss",  min_value=0, max_value=99999, value=0)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Cuba", "Jamaica", "India", "Mexico",
    "South", "Japan", "Philippines", "Germany", "Ecuador",
    "Canada", "England", "China", "Iran", "Taiwan",
    "Italy", "Poland", "Columbia", "Cambodia", "Thailand",
    "Portugal", "Vietnam", "El-Salvador", "Peru",
    "Guatemala", "Dominican-Republic", "Nicaragua",
    "Haiti", "Greece", "Yugoslavia", "Hong", "Ireland",
    "Hungary", "Holand-Netherlands", "France", "Honduras",
    "Trinadad&Tobago", "Scotland", "Laos", "Puerto-Rico",
    "Outlying-US(Guam-USVI-etc)"
])

# ── Build Input DataFrame ──────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "age":            age,
    "workclass":      workclass,
    "fnlwgt":         fnlwgt,
    "education":      education,
    "educational-num": educational_num,
    "marital-status": marital_status,
    "occupation":     occupation,
    "relationship":   relationship,
    "race":           race,
    "gender":         gender,
    "capital-gain":   capital_gain,
    "capital-loss":   capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

# ── Main Panel ─────────────────────────────────────────────────────────────────
st.subheader("📊 Input Summary")
st.dataframe(input_data, use_container_width=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🔍 Predict Income", use_container_width=True, type="primary")

if predict_btn:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("### ✅ Income: **> $50K per year**")
        conf = probability[1] * 100
    else:
        st.warning("### ⚠️ Income: **≤ $50K per year**")
        conf = probability[0] * 100

    # Confidence bar
    st.markdown(f"**Confidence:** {conf:.1f}%")
    st.progress(conf / 100)

    # Probability breakdown
    st.divider()
    st.subheader("📈 Probability Breakdown")
    prob_df = pd.DataFrame({
        "Income Category": ["≤ $50K", "> $50K"],
        "Probability (%)": [round(probability[0]*100, 2), round(probability[1]*100, 2)]
    })
    st.bar_chart(prob_df.set_index("Income Category"))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Model: Random Forest Classifier | Dataset: Adult Income (UCI) | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
