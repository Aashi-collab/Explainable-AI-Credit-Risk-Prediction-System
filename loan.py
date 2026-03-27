import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Loan Prediction App", page_icon="💰", layout="wide")

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💰 Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("User Input")

dependents = st.sidebar.number_input("Dependents", 0, 5)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income = st.sidebar.number_input("Annual Income")
loan_amount = st.sidebar.number_input("Loan Amount")
loan_term = st.sidebar.number_input("Loan Term")
cibil = st.sidebar.number_input("CIBIL Score")

res_assets = st.sidebar.number_input("Residential Assets")
com_assets = st.sidebar.number_input("Commercial Assets")
lux_assets = st.sidebar.number_input("Luxury Assets")
bank_assets = st.sidebar.number_input("Bank Assets")

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Applicant Details")
    st.write(f"Dependents: {dependents}")
    st.write(f"Income: {income}")
    st.write(f"CIBIL Score: {cibil}")

with col2:
    st.subheader("🏦 Asset Details")
    st.write(f"Residential: {res_assets}")
    st.write(f"Commercial: {com_assets}")
    st.write(f"Bank: {bank_assets}")

# Prediction button
st.markdown("---")

if st.button("🚀 Predict Loan Status"):

    total_assets = res_assets + com_assets + lux_assets + bank_assets

    
    input_data = np.array([[dependents, education, self_employed,
                            income, loan_amount, loan_term,
                            cibil, res_assets, com_assets,
                            lux_assets, bank_assets, total_assets]])
    
    input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.markdown("---")
    
    if prediction[0] == 0:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

# Footer
st.markdown("---")
st.markdown("<center>Made by Aashi Jain 💻</center>", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)



    