import joblib
import numpy as np
import pandas as pd

# Define prepare_df
def prepare_df(age, residence_type, loan_purpose, loan_type,
               loan_tenure_months, num_open_accounts, credit_utilization_ratio,
               loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency):
    input_data = {
        'age': age,
        'residence_type': residence_type,
        'loan_purpose': loan_purpose,
        'loan_type': loan_type,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income_ratio,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency
    }
    df = pd.DataFrame([input_data])
    return df



def predict(age, residence_type, loan_purpose, loan_type,
            loan_tenure_months, num_open_accounts, credit_utilization_ratio,
            loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency):
    # Call prepare_df function
    input_df = prepare_df(age, residence_type, loan_purpose, loan_type,
                          loan_tenure_months, num_open_accounts, credit_utilization_ratio,
                          loan_to_income_ratio, delinquency_ratio, avg_dpd_per_delinquency)
    # Call calculate_credit_score function
    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating


# Define calculate_credit_score
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # Transform the input using the pipeline
    pipeline = joblib.load('artifacts/model_credit_risk.joblib')

    coef = pipeline.named_steps['classifier'].coef_
    intercept = pipeline.named_steps['classifier'].intercept_
    input_df_transformed = pipeline.named_steps['preprocessor'].transform(input_df)

    # Ensure the model is a linear model with coefficients
    x = np.dot(input_df_transformed, coef.T) + intercept
    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    # Convert the probability to a credit score
    credit_score = base_score + non_default_probability.flatten() * scale_length
    # Determine the rating category
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
            return 'Undefined'

    rating = get_rating(credit_score[0])
    print(rating)
    return default_probability.flatten()[0], int(credit_score[0]), rating
