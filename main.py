import streamlit as st
from prediction_helper import predict

st.title('Credit Risk Modeling')

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age',min_value=18, max_value=100,step=1)
with row1[1]:
    income = st.number_input('Income',min_value=0, value=200000,step=1)
with row1[2]:
    loan_amount = st.number_input('Loan Amount',min_value=0, value=20000000,step=1)


# Calculate Loan to Income Ratio and display it
loan_to_income_ratio=loan_amount/income if income>0 else 0
with row2[0]:
    st.text('Loan to Income Ratio:')
    st.text(f'{loan_to_income_ratio:.2f}') #Display as text field
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)


with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)


with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])



if st.button('Calculate Risk'):
    probability, credit_score, rating = predict(age, residence_type, loan_purpose, loan_type,
               loan_tenure_months, num_open_accounts,credit_utilization_ratio,
               loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency)

    # Display the results
    st.write(f"Default Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")
