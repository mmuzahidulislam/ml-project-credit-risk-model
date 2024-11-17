
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pipeline = joblib.load('artifacts/model_credit_risk.joblib')
model = pipeline.named_steps['classifier']
def prepare_df(age, residence_type, loan_purpose, loan_type,
               loan_tenure_months, num_open_accounts,credit_utilization_ratio,
               loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency):

    input_data = {
        'age': [age],
        'residence_type':[residence_type],
        'loan_purpose':[loan_purpose],
        'loan_type':[loan_type],
        'loan_tenure_months':[loan_tenure_months],
        'number_of_open_accounts':[num_open_accounts],
        'credit_utilization_ratio':[credit_utilization_ratio],
        'loan_to_income':[loan_to_income_ratio],
        'delinquency_ratio':[delinquency_ratio],
        'avg_dpd_per_delinquency':[avg_dpd_per_delinquency]
    }
    df = pd.DataFrame(input_data)
    return df

def predict(age, residence_type, loan_purpose, loan_type,
               loan_tenure_months, num_open_accounts,credit_utilization_ratio,
               loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency):

    input_df=prepare_df(age, residence_type, loan_purpose, loan_type,
               loan_tenure_months, num_open_accounts,credit_utilization_ratio,
               loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency)

    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating





def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # Assuming 'pipeline' is the preprocessing pipeline used during training
    input_df_transformed = pipeline.transform(input_df)

    x = np.dot(input_df_transformed.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating