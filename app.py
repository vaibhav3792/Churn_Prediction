import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 {
        color: #212529;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    .stProgress > div > div > div > div {
        background-color: #0d6efd;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('artifacts/model.h5')
    return model

@st.cache_resource
def load_encoders():
    with open('artifacts/label_encoder_gender.pkl', 'rb') as file:
        le_gender = pickle.load(file)
    with open('artifacts/one_hot_encoder_geo.pkl', 'rb') as file:
        ohe_geo = pickle.load(file)
    with open('artifacts/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return le_gender, ohe_geo, scaler

model = load_model()
label_encoder_gender, one_hot_encoder_geo, scaler = load_encoders()

col_head1, col_head2, col_head3 = st.columns([1, 6, 1]) # Adjust ratio to make center column wider/narrower
with col_head2:
    st.image("https://miro.medium.com/max/844/1*MyKDLRda6yHGR_8kgVvckg.png", use_container_width=True)

st.title("Customer Retention Analytics")
st.markdown("<p style='text-align: center; color: #6c757d;'>Advanced AI-driven churn risk assessment for banking customers.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics")
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    
    st.subheader("💼 Financials")
    balance = st.number_input('Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=100000.0, step=1000.0)

with col2:
    st.subheader("📊 Account Details")
    credit_score = st.slider('Credit Score', 300, 850, 650)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    
    st.subheader("🔒 Status")
    has_cr_card = st.toggle('Has Credit Card', value=True)
    is_active_member = st.toggle('Is Active Member', value=True)

    has_cr_card = 1 if has_cr_card else 0
    is_active_member = 1 if is_active_member else 0

st.markdown("---")

_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    predict_btn = st.button("Run Risk Assessment")

if predict_btn:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
    
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    st.markdown("### 📋 Risk Analysis Report")
    
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if churn_probability > 0.5:
            st.error(f"⚠️ **High Risk:** This customer is likely to churn.")
        else:
            st.success(f"✅ **Low Risk:** This customer is likely to stay.")

    with res_col2:
        st.metric(label="Churn Probability", value=f"{churn_probability:.2%}")
        st.progress(float(churn_probability))
        
    if churn_probability > 0.5:
        st.warning("💡 **Action:** Schedule a retention call and offer a loyalty bonus immediately.")
    else:
        st.info("💡 **Action:** No immediate action required. Keep engaging via standard newsletters.")