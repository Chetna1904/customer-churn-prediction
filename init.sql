CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    tenure_months     FLOAT,
    monthly_charges   FLOAT,
    total_charges     FLOAT,
    churn_probability FLOAT,
    risk_segment      VARCHAR(20),
    prediction        INTEGER,
    predicted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bulk_predictions (
    id              SERIAL PRIMARY KEY,
    filename        VARCHAR(255),
    total_customers INTEGER,
    high_risk       INTEGER,
    medium_risk     INTEGER,
    low_risk        INTEGER,
    revenue_at_risk FLOAT,
    predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);