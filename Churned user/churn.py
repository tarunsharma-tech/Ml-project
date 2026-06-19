import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#0f172a,
#1e293b,
#0f172a
);
color:white;
}

.hero{
padding:30px;
border-radius:25px;
background:rgba(255,255,255,0.05);
backdrop-filter:blur(15px);
text-align:center;
margin-bottom:20px;
animation:fadein 1s ease;
}

@keyframes fadein{
from{
opacity:0;
transform:translateY(-30px);
}
to{
opacity:1;
transform:translateY(0px);
}
}

.metric-card{
background:rgba(255,255,255,0.05);
padding:20px;
border-radius:20px;
text-align:center;
}

.pred-card{
padding:25px;
border-radius:20px;
text-align:center;
font-size:28px;
font-weight:bold;
}

.stButton>button{
width:100%;
height:60px;
font-size:20px;
font-weight:bold;
border-radius:15px;
background:linear-gradient(
90deg,
#06b6d4,
#3b82f6
);
color:white;
border:none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
<h1>📊 Telco Customer Churn Prediction by Tarun </h1>
<p>Machine Learning Powered Customer Retention Analytics</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------

m1,m2,m3 = st.columns(3)

with m1:
    st.metric("Accuracy","76.21%")

with m2:
    st.metric("F1 Score","65.57%")

with m3:
    st.metric("Model","Logistic Regression")

st.divider()

# ---------------- INPUTS ----------------

st.subheader("Customer Information")

c1,c2,c3 = st.columns(3)

with c1:
    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        [0,1]
    )

with c2:
    dependents = st.selectbox(
        "Dependents",
        [0,1]
    )

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    phone_service = st.selectbox(
        "Phone Service",
        [0,1]
    )

with c3:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        [0,1]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [0,1]
    )

    monthly = st.number_input(
        "Monthly Charges",
        value=70.0
    )

# ---------------- SECOND ROW ----------------

c4,c5,c6 = st.columns(3)

with c4:
    internet = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    online_security = st.selectbox(
        "Online Security",
        [0,1]
    )

with c5:
    online_backup = st.selectbox(
        "Online Backup",
        [0,1]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [0,1]
    )

with c6:
    tech_support = st.selectbox(
        "Tech Support",
        [0,1]
    )

    stream_tv = st.selectbox(
        "Streaming TV",
        [0,1]
    )

# ---------------- THIRD ROW ----------------

c7,c8,c9 = st.columns(3)

with c7:
    stream_movies = st.selectbox(
        "Streaming Movies",
        [0,1]
    )

with c8:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with c9:
    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

total = st.number_input(
    "Total Charges",
    value=1000.0
)

# ---------------- MAPPINGS ----------------

gender_map = {
    "Female":0,
    "Male":1
}

internet_map = {
    "DSL":0,
    "Fiber optic":1,
    "No":2
}

contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

payment_map = {
    "Bank transfer (automatic)":0,
    "Credit card (automatic)":1,
    "Electronic check":2,
    "Mailed check":3
}

# ---------------- PREDICT ----------------

if st.button("🚀 Predict Churn"):

    data = pd.DataFrame([[
        gender_map[gender],
        senior,
        partner,
        dependents,
        tenure,
        phone_service,
        multiple_lines,
        internet_map[internet],
        online_security,
        online_backup,
        device_protection,
        tech_support,
        stream_tv,
        stream_movies,
        contract_map[contract],
        paperless,
        payment_map[payment],
        monthly,
        total
    ]],
    columns=[
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'tenure',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges'
    ])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"🔴 High Churn Risk ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"🟢 Customer Likely To Stay ({(1-probability)*100:.2f}%)"
        )

    st.subheader("Risk Meter")

    st.progress(float(probability))

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.snow()

st.divider()

st.caption(
"Built by Tarun sharma with Streamlit • Logistic Regression • Telco Churn Dataset"
)