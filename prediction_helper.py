import numpy as np
import pandas as pd
import joblib

# Load the trained model and related objects
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
col_to_scale = model_data['col_to_scale']


# ==============================
# 1️⃣ Prepare input function
# ==============================
def prepare_input(age, income, loan_amount, loan_tenure_months,
                  avg_dpd_per_deliquent, delinquent_months_ratio,
                  credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    # Create dictionary with all required model features
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income_ratio': loan_amount / income if income > 0 else 0,
        'delinquent_months_ratio': delinquent_months_ratio,
        'avg_dpd_per_deliquent': avg_dpd_per_deliquent,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy values for non-user input fields
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    # Build DataFrame
    df = pd.DataFrame([input_data])

    # Apply scaling to numeric columns
    df[col_to_scale] = scaler.transform(df[col_to_scale])

    # Ensure same feature order as training
    df = df[features]

    return df


# ==============================
# 2️⃣ Prediction + scoring
# ==============================
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

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
    return default_probability.flatten()[0], int(credit_score[0]), rating


# ==============================
# 3️⃣ Predict wrapper function
# ==============================
def predict(age, income, loan_amount, loan_tenure_months,
            avg_dpd_per_deliquent, delinquent_months_ratio,
            credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months,
                             avg_dpd_per_deliquent, delinquent_months_ratio,
                             credit_utilization_ratio, num_open_accounts,
                             residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating
