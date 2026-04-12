# ================================
# 1. Import Libraries
# ================================
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import joblib
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import inch
from datetime import datetime


# ================================
# 2. Page Configuration
# ================================
st.set_page_config(
    page_title = "Customer Churn Prediction System",
    page_icon  = "📊",
    layout     = "wide"
)


# ================================
# 3. Load Model
# ================================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")
    return joblib.load(model_path)

model = load_model()


# ================================
# 4. Constants
# ================================
API_URL = "http://127.0.0.1:8000"

FEATURE_COLUMNS = [
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

SCALE_PARAMS = {
    'Tenure Months'  : {'mean': 32.4,   'std': 24.6},
    'Monthly Charges': {'mean': 64.8,   'std': 30.1},
    'Total Charges'  : {'mean': 2283.3, 'std': 2266.8},
}


# ================================
# 5. Styling
# ================================
st.markdown("""
<style>
.main-title    { font-size: 2rem; font-weight: 700; color: #1D9E75; }
.sub-title     { font-size: 1rem; color: #888; margin-bottom: 1rem; }
.high-risk     { border-left: 5px solid #E24B4A; background: #fff5f5;
                 padding: 16px; border-radius: 8px; margin-bottom: 10px; }
.med-risk      { border-left: 5px solid #EF9F27; background: #fffbf0;
                 padding: 16px; border-radius: 8px; margin-bottom: 10px; }
.low-risk      { border-left: 5px solid #1D9E75; background: #f0fff8;
                 padding: 16px; border-radius: 8px; margin-bottom: 10px; }
.strategy-box  { background: #f0fff8; border-left: 4px solid #1D9E75;
                 padding: 10px 14px; border-radius: 8px;
                 margin: 5px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ================================
# 6. Helper Functions
# ================================
def get_risk_segment(prob):
    if prob >= 0.7:   return "High Risk"
    elif prob >= 0.4: return "Medium Risk"
    else:             return "Low Risk"


def get_strategies(row):
    strategies = []
    if row.get('Tenure Months', 0) < 0:
        strategies.append("Assign dedicated onboarding manager for first 6 months")
        strategies.append("Offer welcome loyalty package — free month or upgrade")
    if row.get('Internet Service_Fiber optic', 0) == 1:
        strategies.append("Offer fiber service quality review and speed guarantee")
        strategies.append("Provide fiber loyalty discount — 10% off for 6 months")
    if row.get('Monthly Charges', 0) > 0:
        strategies.append("Offer customized bundle plan to reduce monthly bill")
        strategies.append("Provide loyalty discount — 15% off next 3 months")
    if row.get('Dependents_Yes', 0) == 0:
        strategies.append("Promote family/dependents plan benefits and discounts")
    if row.get('Payment Method_Electronic check', 0) == 1:
        strategies.append("Incentivize switch to auto-pay — offer 5% discount")
        strategies.append("Send email campaign highlighting auto-pay convenience")
    if row.get('Contract_Two year', 0) == 0 and row.get('Contract_One year', 0) == 0:
        strategies.append("Offer discounted 1-year or 2-year contract upgrade")
        strategies.append("Highlight contract benefits — price lock and priority support")
    if row.get('Multiple Lines_Yes', 0) == 0:
        strategies.append("Promote multi-line plan — offer second line at 50% off")
    if row.get('Streaming TV_Yes', 0) == 0 and row.get('Streaming Movies_Yes', 0) == 0:
        strategies.append("Offer free 3-month streaming TV and movies trial")
    if not strategies:
        strategies.append("Customer is stable — send quarterly satisfaction survey")
    return " | ".join(strategies)


def scale_features(df):
    df = df.copy()
    for col, params in SCALE_PARAMS.items():
        if col in df.columns:
            df[col] = (df[col] - params['mean']) / params['std']
    return df


def preprocess_uploaded_csv(df):
    processed = pd.DataFrame()

    # Numerical features
    for col in ['Tenure Months', 'Monthly Charges', 'Total Charges']:
        if col in df.columns:
            processed[col] = pd.to_numeric(
                df[col], errors='coerce'
            ).fillna(0)
        else:
            processed[col] = 0

    # Binary features
    binary_map = {
        'Gender_Male'          : ('gender',           'Male'),
        'Senior Citizen_Yes'   : ('Senior Citizen',   '1'),
        'Partner_Yes'          : ('Partner',          'Yes'),
        'Dependents_Yes'       : ('Dependents',       'Yes'),
        'Phone Service_Yes'    : ('Phone Service',    'Yes'),
        'Paperless Billing_Yes': ('Paperless Billing','Yes'),
    }
    for feat, (col, val) in binary_map.items():
        if col in df.columns:
            processed[feat] = (
                df[col].astype(str).str.strip() == val
            ).astype(int)
        else:
            processed[feat] = 0

    # Multiple Lines
    if 'Multiple Lines' in df.columns:
        processed['Multiple Lines_No phone service'] = (
            df['Multiple Lines'] == 'No phone service').astype(int)
        processed['Multiple Lines_Yes'] = (
            df['Multiple Lines'] == 'Yes').astype(int)
    else:
        processed['Multiple Lines_No phone service'] = 0
        processed['Multiple Lines_Yes']              = 0

    # Internet Service
    if 'Internet Service' in df.columns:
        processed['Internet Service_Fiber optic'] = (
            df['Internet Service'] == 'Fiber optic').astype(int)
        processed['Internet Service_No'] = (
            df['Internet Service'] == 'No').astype(int)
    else:
        processed['Internet Service_Fiber optic'] = 0
        processed['Internet Service_No']          = 0

    # Service features
    service_map = {
        'Online Security' : 'Online Security',
        'Online Backup'   : 'Online Backup',
        'Device Protection': 'Device Protection',
        'Tech Support'    : 'Tech Support',
        'Streaming TV'    : 'Streaming TV',
        'Streaming Movies': 'Streaming Movies',
    }
    for feat_prefix, col in service_map.items():
        if col in df.columns:
            processed[f'{feat_prefix}_No internet service'] = (
                df[col] == 'No internet service').astype(int)
            processed[f'{feat_prefix}_Yes'] = (
                df[col] == 'Yes').astype(int)
        else:
            processed[f'{feat_prefix}_No internet service'] = 0
            processed[f'{feat_prefix}_Yes']                 = 0

    # Contract
    if 'Contract' in df.columns:
        processed['Contract_One year'] = (
            df['Contract'] == 'One year').astype(int)
        processed['Contract_Two year'] = (
            df['Contract'] == 'Two year').astype(int)
    else:
        processed['Contract_One year'] = 0
        processed['Contract_Two year'] = 0

    # Payment Method
    if 'Payment Method' in df.columns:
        processed['Payment Method_Credit card (automatic)'] = (
            df['Payment Method'] == 'Credit card (automatic)').astype(int)
        processed['Payment Method_Electronic check'] = (
            df['Payment Method'] == 'Electronic check').astype(int)
        processed['Payment Method_Mailed check'] = (
            df['Payment Method'] == 'Mailed check').astype(int)
    else:
        processed['Payment Method_Credit card (automatic)'] = 0
        processed['Payment Method_Electronic check']        = 0
        processed['Payment Method_Mailed check']            = 0

    # Scale numerical
    processed = scale_features(processed)

    # Ensure correct column order
    for col in FEATURE_COLUMNS:
        if col not in processed.columns:
            processed[col] = 0

    return processed[FEATURE_COLUMNS]


def run_bulk_predictions(df_raw):
    X_processed = preprocess_uploaded_csv(df_raw)
    probs       = model.predict_proba(X_processed)[:, 1]
    segments    = [get_risk_segment(p) for p in probs]
    strategies  = [
        get_strategies(row)
        for _, row in X_processed.iterrows()
    ]

    df_result = df_raw.copy()
    df_result['Churn_Probability'] = np.round(probs, 4)
    df_result['Risk_Segment']      = segments
    df_result['Retention_Strategy']= strategies

    # Default group = segment
    df_result['Customer_Group'] = df_result['Risk_Segment']

    # Cluster high risk customers into subgroups
    df_high = df_result[df_result['Risk_Segment'] == 'High Risk'].copy()
    if len(df_high) >= 3:
        X_high  = preprocess_uploaded_csv(df_high)
        n_clust = min(3, len(df_high))
        kmeans  = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels  = kmeans.fit_predict(X_high) + 1
        df_result.loc[df_high.index, 'Customer_Group'] = [
            f"High Risk Group {l}" for l in labels
        ]

    return df_result


def generate_pdf_report(summary_stats, top_reasons, group_strategies):
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50,  bottomMargin=50
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent    = styles['Title'],
        fontSize  = 22,
        textColor = colors.HexColor('#1D9E75'),
        spaceAfter= 6
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent    = styles['Heading2'],
        fontSize  = 14,
        textColor = colors.HexColor('#185FA5'),
        spaceAfter= 6
    )

    story = []

    # Title
    story.append(Paragraph(
        "Customer Churn Prediction Report", title_style
    ))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles['Normal']
    ))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#1D9E75')
    ))
    story.append(Spacer(1, 0.2*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        f"This report analyzes <b>{summary_stats['total']:,}</b> customers "
        f"using an XGBoost ML model (Recall: 84%, ROC-AUC: 83.3%). "
        f"Identified <b>{summary_stats['high']:,} high risk</b>, "
        f"<b>{summary_stats['medium']:,} medium risk</b>, and "
        f"<b>{summary_stats['low']:,} low risk</b> customers. "
        f"Total monthly revenue at risk: "
        f"<b>${summary_stats['revenue']:,}</b>.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))

    # Key Metrics Table
    story.append(Paragraph("Key Metrics", heading_style))
    table_data = [
        ['Metric', 'Value'],
        ['Total Customers',
         f"{summary_stats['total']:,}"],
        ['High Risk Customers',
         f"{summary_stats['high']:,} ({summary_stats['high_pct']:.1f}%)"],
        ['Medium Risk Customers',
         f"{summary_stats['medium']:,} ({summary_stats['med_pct']:.1f}%)"],
        ['Low Risk Customers',
         f"{summary_stats['low']:,} ({summary_stats['low_pct']:.1f}%)"],
        ['Average Churn Probability',
         f"{summary_stats['avg_prob']:.1%}"],
        ['Monthly Revenue at Risk',
         f"${summary_stats['revenue']:,}"],
    ]
    t = Table(table_data, colWidths=[3.5*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D9E75')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # Top Churn Reasons
    story.append(Paragraph("Top Reasons Customers Are Churning", heading_style))
    reasons_data = [['Rank', 'Churn Factor', 'Impact Level']]
    for i, (reason, impact) in enumerate(top_reasons, 1):
        reasons_data.append([str(i), reason, impact])
    r = Table(reasons_data, colWidths=[0.5*inch, 4*inch, 2*inch])
    r.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#185FA5')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f0f7ff'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(r)
    story.append(Spacer(1, 0.3*inch))

    # Group Strategies
    story.append(Paragraph(
        "Customer Group Retention Strategies", heading_style
    ))
    for group_name, strats, count in group_strategies:
        story.append(Paragraph(
            f"<b>{group_name}</b> ({count:,} customers)",
            styles['Heading3']
        ))
        for s in strats:
            story.append(Paragraph(f"• {s}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=colors.grey
    ))
    story.append(Paragraph(
        "Generated automatically by the Customer Churn Prediction System.",
        styles['Italic']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================================
# 7. Sample CSV Data
# ================================
def get_sample_csv():
    data = {
        'customerID'       : ['CUST001','CUST002','CUST003','CUST004','CUST005'],
        'gender'           : ['Male','Female','Male','Female','Male'],
        'Senior Citizen'   : [0, 0, 1, 0, 0],
        'Partner'          : ['Yes','No','No','Yes','No'],
        'Dependents'       : ['No','Yes','No','No','Yes'],
        'Tenure Months'    : [2, 45, 12, 60, 5],
        'Phone Service'    : ['Yes','Yes','No','Yes','Yes'],
        'Multiple Lines'   : ['No','Yes','No phone service','Yes','No'],
        'Internet Service' : ['Fiber optic','DSL','Fiber optic','DSL','Fiber optic'],
        'Online Security'  : ['No','Yes','No','Yes','No'],
        'Online Backup'    : ['No','Yes','No','Yes','No'],
        'Device Protection': ['No','Yes','No','Yes','No'],
        'Tech Support'     : ['No','Yes','No','Yes','No'],
        'Streaming TV'     : ['Yes','No','Yes','No','Yes'],
        'Streaming Movies' : ['Yes','No','No','No','Yes'],
        'Contract'         : ['Month-to-month','Two year','Month-to-month',
                              'Two year','Month-to-month'],
        'Paperless Billing': ['Yes','No','Yes','No','Yes'],
        'Payment Method'   : ['Electronic check','Credit card (automatic)',
                              'Electronic check','Bank transfer (automatic)',
                              'Electronic check'],
        'Monthly Charges'  : [85.0, 45.0, 70.0, 42.0, 90.0],
        'Total Charges'    : [170.0, 2025.0, 840.0, 2520.0, 450.0],
    }
    return pd.DataFrame(data)


# ================================
# 8. App Header
# ================================
st.markdown(
    '<p class="main-title">📊 Customer Churn Prediction System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Powered by XGBoost · Recall 84% · ROC-AUC 83.3%</p>',
    unsafe_allow_html=True
)

page = st.radio(
    "Select Mode",
    ["👤 Single Customer", "📁 Bulk CSV Upload", "📑 Report Generator"],
    horizontal=True
)
st.divider()


# ================================
# PAGE 1 — Single Customer
# ================================
if page == "👤 Single Customer":

    st.subheader("Single Customer Churn Prediction")

    # Sidebar inputs
    st.sidebar.title("👤 Customer Details")
    st.sidebar.subheader("📋 Basic Info")
    tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges  = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges    = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen   = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner          = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents       = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    st.sidebar.subheader("🌐 Services")
    phone_service    = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines   = st.sidebar.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"])
    internet         = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.sidebar.selectbox(
        "Online Security", ["No", "Yes", "No internet service"])
    online_backup    = st.sidebar.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"])
    device_prot      = st.sidebar.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"])
    tech_support     = st.sidebar.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv     = st.sidebar.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"])
    st.sidebar.subheader("📄 Contract & Billing")
    contract         = st.sidebar.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless        = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment          = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    predict_btn = st.sidebar.button(
        "🔮 Predict Churn", type="primary", use_container_width=True
    )

    if predict_btn:
        payload = {
            "tenure_months"               : (tenure - 32.4) / 24.6,
            "monthly_charges"             : (monthly_charges - 64.8) / 30.1,
            "total_charges"               : (total_charges - 2283.3) / 2266.8,
            "gender_male"                 : 1 if gender == "Male" else 0,
            "senior_citizen"              : 1 if senior_citizen == "Yes" else 0,
            "partner"                     : 1 if partner == "Yes" else 0,
            "dependents"                  : 1 if dependents == "Yes" else 0,
            "phone_service"               : 1 if phone_service == "Yes" else 0,
            "multiple_lines_no_phone"     : 1 if multiple_lines == "No phone service" else 0,
            "multiple_lines_yes"          : 1 if multiple_lines == "Yes" else 0,
            "internet_fiber"              : 1 if internet == "Fiber optic" else 0,
            "internet_no"                 : 1 if internet == "No" else 0,
            "online_security_no_internet" : 1 if online_security == "No internet service" else 0,
            "online_security_yes"         : 1 if online_security == "Yes" else 0,
            "online_backup_no_internet"   : 1 if online_backup == "No internet service" else 0,
            "online_backup_yes"           : 1 if online_backup == "Yes" else 0,
            "device_protection_no_internet":1 if device_prot == "No internet service" else 0,
            "device_protection_yes"       : 1 if device_prot == "Yes" else 0,
            "tech_support_no_internet"    : 1 if tech_support == "No internet service" else 0,
            "tech_support_yes"            : 1 if tech_support == "Yes" else 0,
            "streaming_tv_no_internet"    : 1 if streaming_tv == "No internet service" else 0,
            "streaming_tv_yes"            : 1 if streaming_tv == "Yes" else 0,
            "streaming_movies_no_internet": 1 if streaming_movies == "No internet service" else 0,
            "streaming_movies_yes"        : 1 if streaming_movies == "Yes" else 0,
            "contract_one_year"           : 1 if contract == "One year" else 0,
            "contract_two_year"           : 1 if contract == "Two year" else 0,
            "paperless_billing"           : 1 if paperless == "Yes" else 0,
            "payment_credit_card"         : 1 if payment == "Credit card (automatic)" else 0,
            "payment_electronic_check"    : 1 if payment == "Electronic check" else 0,
            "payment_mailed_check"        : 1 if payment == "Mailed check" else 0,
        }

        with st.spinner("Analyzing customer..."):
            try:
                response   = requests.post(f"{API_URL}/predict", json=payload)
                result     = response.json()
                prob       = result['churn_probability']
                segment    = result['risk_segment']
                strategies = result['retention_strategies']

                color_map = {
                    "High Risk"  : "high-risk",
                    "Medium Risk": "med-risk",
                    "Low Risk"   : "low-risk"
                }
                icon_map = {
                    "High Risk": "🔴",
                    "Medium Risk": "🟡",
                    "Low Risk": "🟢"
                }

                st.markdown(
                    f'<div class="{color_map[segment]}">'
                    f'<h2>{icon_map[segment]} {segment.upper()} CUSTOMER</h2>'
                    f'<p>{result["summary"]}</p></div>',
                    unsafe_allow_html=True
                )
                st.divider()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Churn Probability", f"{round(prob*100,1)}%")
                m2.metric("Risk Segment",       segment)
                m3.metric("Prediction",
                          "Will Churn" if result['churn_prediction'] == 1
                          else "Will Stay")
                m4.metric("Strategies Generated", len(strategies))
                st.divider()

                left, right = st.columns(2)
                with left:
                    st.subheader("🎯 Retention Strategies")
                    for s in strategies:
                        st.markdown(
                            f'<div class="strategy-box">→ {s}</div>',
                            unsafe_allow_html=True
                        )
                with right:
                    st.subheader("📊 Risk Gauge")
                    fig = go.Figure(go.Indicator(
                        mode  = "gauge+number",
                        value = round(prob * 100, 1),
                        title = {'text': "Churn Probability %"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar' : {'color': (
                                "#E24B4A" if prob >= 0.7 else
                                "#EF9F27" if prob >= 0.4 else
                                "#1D9E75"
                            )},
                            'steps': [
                                {'range': [0,  40], 'color': '#EAF3DE'},
                                {'range': [40, 70], 'color': '#FAEEDA'},
                                {'range': [70,100], 'color': '#FCEBEB'},
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                shap_path = os.path.join(
                    os.path.dirname(__file__),
                    "model", "shap_summary_bar.png"
                )
                if os.path.exists(shap_path):
                    st.divider()
                    st.subheader("🔍 Feature Importance (SHAP)")
                    st.image(shap_path, use_container_width=True)

            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("Make sure FastAPI is running: uvicorn app:app --reload")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", "7,043")
        c2.metric("Churn Rate",      "26.5%")
        c3.metric("High Risk",       "2,487")
        c4.metric("Revenue at Risk", "$224,445/mo")

        shap_path = os.path.join(
            os.path.dirname(__file__), "model", "shap_summary_bar.png"
        )
        if os.path.exists(shap_path):
            st.divider()
            st.subheader("🔍 Top Factors Driving Churn")
            st.image(shap_path, use_container_width=True)


# ================================
# PAGE 2 — Bulk CSV Upload
# ================================
elif page == "📁 Bulk CSV Upload":

    st.subheader("📁 Bulk Customer Churn Prediction")
    st.markdown(
        "Upload your customer CSV and get predictions for "
        "all customers instantly."
    )

    # Sample CSV download
    sample_csv = get_sample_csv().to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Sample CSV Template",
        sample_csv,
        "sample_customers.csv",
        "text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Customer CSV File",
        type=['csv']
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success(
            f"✅ File uploaded — {len(df_raw):,} customers found"
        )

        with st.expander("Preview uploaded data (first 10 rows)"):
            st.dataframe(df_raw.head(10), use_container_width=True)

        if st.button("🔮 Run Bulk Prediction", type="primary"):
            with st.spinner(
                f"Running predictions for {len(df_raw):,} customers..."
            ):
                df_result = run_bulk_predictions(df_raw)

            st.success("✅ Predictions complete!")
            st.divider()

            # Summary metrics
            high_count = int((df_result['Risk_Segment'] == 'High Risk').sum())
            med_count  = int((df_result['Risk_Segment'] == 'Medium Risk').sum())
            low_count  = int((df_result['Risk_Segment'] == 'Low Risk').sum())
            total      = len(df_result)
            rev_risk   = (high_count + med_count) * 65

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Customers", f"{total:,}")
            m2.metric("🔴 High Risk",    f"{high_count:,}",
                      delta=f"{high_count/total*100:.1f}%",
                      delta_color="inverse")
            m3.metric("🟡 Medium Risk",  f"{med_count:,}")
            m4.metric("🟢 Low Risk",     f"{low_count:,}",
                      delta=f"{low_count/total*100:.1f}%")
            m5.metric("Revenue at Risk", f"${rev_risk:,}")

            st.divider()

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=[high_count, med_count, low_count],
                    names=['High Risk', 'Medium Risk', 'Low Risk'],
                    color_discrete_sequence=[
                        '#E24B4A', '#EF9F27', '#1D9E75'
                    ],
                    title="Risk Segment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    df_result,
                    x='Churn_Probability',
                    color='Risk_Segment',
                    color_discrete_map={
                        'High Risk'  : '#E24B4A',
                        'Medium Risk': '#EF9F27',
                        'Low Risk'   : '#1D9E75'
                    },
                    title="Churn Probability Distribution",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)

            # Customer Groups
            st.divider()
            st.subheader("👥 Customer Groups & Strategies")

            groups = df_result['Customer_Group'].value_counts()
            for group, count in groups.items():
                grp_df    = df_result[df_result['Customer_Group'] == group]
                avg_prob  = grp_df['Churn_Probability'].mean()
                sample_st = grp_df['Retention_Strategy'].iloc[0].split(" | ")

                if "High" in str(group):
                    bg, border = "#fff5f5", "#E24B4A"
                elif "Medium" in str(group):
                    bg, border = "#fffbf0", "#EF9F27"
                else:
                    bg, border = "#f0fff8", "#1D9E75"

                st.markdown(
                    f'<div style="background:{bg};border-left:4px solid '
                    f'{border};padding:14px;border-radius:8px;margin:8px 0">'
                    f'<b>{group}</b> — {count:,} customers | '
                    f'Avg churn probability: {avg_prob:.1%}</div>',
                    unsafe_allow_html=True
                )
                with st.expander(f"View strategies for {group}"):
                    for s in sample_st[:6]:
                        if s.strip():
                            st.markdown(
                                f'<div class="strategy-box">→ {s}</div>',
                                unsafe_allow_html=True
                            )

            # Download
            st.divider()
            csv_output = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Segmented Customer CSV",
                csv_output,
                "churn_predictions.csv",
                "text/csv",
                use_container_width=True
            )


# ================================
# PAGE 3 — Report Generator
# ================================
elif page == "📑 Report Generator":

    st.subheader("📑 Automated Business Report Generator")
    st.markdown(
        "Upload your customer CSV to generate a complete "
        "business PDF report automatically."
    )

    sample_csv = get_sample_csv().to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Sample CSV Template",
        sample_csv,
        "sample_customers.csv",
        "text/csv"
    )

    st.divider()

    uploaded_report = st.file_uploader(
        "Upload Customer CSV File",
        type=['csv'],
        key="report_uploader"
    )

    if uploaded_report is not None:
        df_raw = pd.read_csv(uploaded_report)
        st.success(f"✅ {len(df_raw):,} customers loaded")

        if st.button("📑 Generate Full Report", type="primary"):
            with st.spinner("Generating report..."):

                df_result  = run_bulk_predictions(df_raw)
                probs      = df_result['Churn_Probability'].values
                high_count = int((df_result['Risk_Segment'] == 'High Risk').sum())
                med_count  = int((df_result['Risk_Segment'] == 'Medium Risk').sum())
                low_count  = int((df_result['Risk_Segment'] == 'Low Risk').sum())
                total      = len(df_result)

                summary_stats = {
                    'total'   : total,
                    'high'    : high_count,
                    'medium'  : med_count,
                    'low'     : low_count,
                    'high_pct': high_count / total * 100,
                    'med_pct' : med_count  / total * 100,
                    'low_pct' : low_count  / total * 100,
                    'avg_prob': float(probs.mean()),
                    'revenue' : (high_count + med_count) * 65,
                }

                top_reasons = [
                    ("Low Tenure — new customers not yet committed",  "Very High"),
                    ("Fiber Optic Internet — high cost perception",   "High"),
                    ("High Monthly Charges — price sensitivity",      "High"),
                    ("No Dependents — single users switch easily",    "Medium"),
                    ("Electronic Check Payment — less committed",     "Medium"),
                    ("Month-to-Month Contract — no lock-in",         "High"),
                ]

                group_strategies = [
                    (
                        "High Risk — New Fiber Optic Customers",
                        ["Assign dedicated onboarding manager",
                         "Fiber loyalty discount 10% off for 6 months",
                         "Service quality review and speed guarantee",
                         "Offer contract upgrade with price lock"],
                        high_count // 2
                    ),
                    (
                        "High Risk — High Charge Month-to-Month Customers",
                        ["Offer customized bundle to reduce monthly bill",
                         "Provide 15% loyalty discount for 3 months",
                         "Incentivize switch to annual contract"],
                        high_count - high_count // 2
                    ),
                    (
                        "Medium Risk Customers",
                        ["Send personalized retention email",
                         "Offer service upgrade trial",
                         "Promote family plan benefits"],
                        med_count
                    ),
                    (
                        "Low Risk — Loyal Customers",
                        ["Send quarterly satisfaction survey",
                         "Offer multi-line plan promotion",
                         "Promote streaming services trial"],
                        low_count
                    ),
                ]

            # Preview in app
            st.divider()
            st.subheader("📊 Report Preview")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Customers",  f"{total:,}")
            p2.metric("High Risk",        f"{high_count:,}")
            p3.metric("Revenue at Risk",  f"${summary_stats['revenue']:,}")
            p4.metric("Avg Churn Prob",   f"{summary_stats['avg_prob']:.1%}")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=[high_count, med_count, low_count],
                    names=['High Risk', 'Medium Risk', 'Low Risk'],
                    color_discrete_sequence=['#E24B4A','#EF9F27','#1D9E75'],
                    title="Customer Risk Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    x=['High Risk', 'Medium Risk', 'Low Risk'],
                    y=[high_count, med_count, low_count],
                    color=['High Risk', 'Medium Risk', 'Low Risk'],
                    color_discrete_map={
                        'High Risk'  : '#E24B4A',
                        'Medium Risk': '#EF9F27',
                        'Low Risk'   : '#1D9E75'
                    },
                    title="Customer Count by Risk Segment"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📋 Top Churn Reasons")
            for i, (reason, impact) in enumerate(top_reasons, 1):
                color = ("#FCEBEB" if impact == "Very High" else
                         "#fff5f5" if impact == "High" else "#fffbf0")
                st.markdown(
                    f'<div style="background:{color};padding:10px 14px;'
                    f'border-radius:8px;margin:5px 0">'
                    f'<b>{i}.</b> {reason} — '
                    f'<b>Impact: {impact}</b></div>',
                    unsafe_allow_html=True
                )

            # Generate PDF
            st.divider()
            pdf_buffer = generate_pdf_report(
                summary_stats, top_reasons, group_strategies
            )
            st.download_button(
                "⬇️ Download Full PDF Report",
                pdf_buffer,
                f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                "application/pdf",
                use_container_width=True
            )
            st.success(
                "✅ Report ready! Click the button above to download."
            )