# ================================
# 1. Import Libraries
# ================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os


# ================================
# 2. Initialize App
# ================================
app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = "Predicts customer churn probability and provides retention strategies",
    version     = "1.0.0"
)

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ================================
# 3. Load Model on Startup
# ================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")
model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model).__name__}")


# ================================
# 4. Define Input Schema
# (Matches your exact feature columns)
# ================================
class CustomerData(BaseModel):
    tenure_months              : float
    monthly_charges            : float
    total_charges              : float
    gender_male                : int
    senior_citizen             : int
    partner                    : int
    dependents                 : int
    phone_service              : int
    multiple_lines_no_phone    : int
    multiple_lines_yes         : int
    internet_fiber             : int
    internet_no                : int
    online_security_no_internet: int
    online_security_yes        : int
    online_backup_no_internet  : int
    online_backup_yes          : int
    device_protection_no_internet: int
    device_protection_yes      : int
    tech_support_no_internet   : int
    tech_support_yes           : int
    streaming_tv_no_internet   : int
    streaming_tv_yes           : int
    streaming_movies_no_internet: int
    streaming_movies_yes       : int
    contract_one_year          : int
    contract_two_year          : int
    paperless_billing          : int
    payment_credit_card        : int
    payment_electronic_check   : int
    payment_mailed_check       : int


# ================================
# 5. Retention Strategy Function
# ================================
def get_retention_strategy(data: CustomerData):
    strategies = []

    if data.tenure_months < 0:
        strategies.append("Assign dedicated onboarding manager for first 6 months")
        strategies.append("Offer welcome loyalty package — free month or service upgrade")

    if data.internet_fiber == 1:
        strategies.append("Offer fiber optic service quality review and speed guarantee")
        strategies.append("Provide fiber loyalty discount — 10% off for 6 months")

    if data.monthly_charges > 0:
        strategies.append("Offer customized bundle plan to reduce monthly bill")
        strategies.append("Provide loyalty discount — 15% off next 3 months")

    if data.dependents == 0:
        strategies.append("Promote family/dependents plan benefits and discounts")

    if data.payment_electronic_check == 1:
        strategies.append("Incentivize switch to auto-pay — offer 5% monthly discount")
        strategies.append("Send email campaign highlighting auto-pay convenience")

    if data.contract_two_year == 0 and data.contract_one_year == 0:
        strategies.append("Offer discounted 1-year or 2-year contract upgrade")
        strategies.append("Highlight contract benefits — price lock and priority support")

    if data.multiple_lines_yes == 0:
        strategies.append("Promote multi-line plan — offer second line at 50% off")

    if data.streaming_tv_yes == 0 and data.streaming_movies_yes == 0:
        strategies.append("Offer free 3-month streaming TV and movies trial")

    if not strategies:
        strategies.append("Customer profile is stable — send quarterly satisfaction survey")

    return strategies


# ================================
# 6. Risk Segment Function
# ================================
def get_risk_segment(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


# ================================
# 7. Feature Order
# (Must match exact order from training)
# ================================
FEATURE_ORDER = [
    'Tenure Months', 'Monthly Charges', 'Total Charges',
    'Gender_Male', 'Senior Citizen_Yes', 'Partner_Yes', 'Dependents_Yes',
    'Phone Service_Yes', 'Multiple Lines_No phone service', 'Multiple Lines_Yes',
    'Internet Service_Fiber optic', 'Internet Service_No',
    'Online Security_No internet service', 'Online Security_Yes',
    'Online Backup_No internet service', 'Online Backup_Yes',
    'Device Protection_No internet service', 'Device Protection_Yes',
    'Tech Support_No internet service', 'Tech Support_Yes',
    'Streaming TV_No internet service', 'Streaming TV_Yes',
    'Streaming Movies_No internet service', 'Streaming Movies_Yes',
    'Contract_One year', 'Contract_Two year',
    'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check',
    'Payment Method_Mailed check'
]


# ================================
# 8. API Endpoints
# ================================

# -- Home endpoint --
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "version": "1.0.0",
        "endpoints": {
            "predict" : "/predict",
            "health"  : "/health",
            "docs"    : "/docs"
        }
    }


# -- Health check --
@app.get("/health")
def health():
    return {
        "status"    : "healthy",
        "model_type": type(model).__name__
    }


# -- Main prediction endpoint --
@app.post("/predict")
def predict(customer: CustomerData):

    # Build feature array in correct order
    features = pd.DataFrame([[
        customer.tenure_months,
        customer.monthly_charges,
        customer.total_charges,
        customer.gender_male,
        customer.senior_citizen,
        customer.partner,
        customer.dependents,
        customer.phone_service,
        customer.multiple_lines_no_phone,
        customer.multiple_lines_yes,
        customer.internet_fiber,
        customer.internet_no,
        customer.online_security_no_internet,
        customer.online_security_yes,
        customer.online_backup_no_internet,
        customer.online_backup_yes,
        customer.device_protection_no_internet,
        customer.device_protection_yes,
        customer.tech_support_no_internet,
        customer.tech_support_yes,
        customer.streaming_tv_no_internet,
        customer.streaming_tv_yes,
        customer.streaming_movies_no_internet,
        customer.streaming_movies_yes,
        customer.contract_one_year,
        customer.contract_two_year,
        customer.paperless_billing,
        customer.payment_credit_card,
        customer.payment_electronic_check,
        customer.payment_mailed_check,
    ]], columns=FEATURE_ORDER)

    # Get prediction
    churn_probability = float(model.predict_proba(features)[0][1])
    churn_prediction  = int(churn_probability >= 0.5)
    risk_segment      = get_risk_segment(churn_probability)
    strategies        = get_retention_strategy(customer)

    return {
        "churn_prediction"  : churn_prediction,
        "churn_probability" : round(churn_probability, 4),
        "risk_segment"      : risk_segment,
        "retention_strategies": strategies,
        "summary": f"This customer has a {round(churn_probability * 100, 1)}% chance of churning and is classified as {risk_segment}."
    }


# ================================
# 9. Run the App
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)