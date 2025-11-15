import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# Set plot style
sns.set(style="whitegrid")

# -----------------------------
# STEP 1 — Load Model & Scaler
# -----------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# STEP 2 — Load Dataset
# -----------------------------
data = pd.read_csv("breast_cancer_data.csv")
data = data.drop(columns=['Unnamed: 32', 'id'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Extract feature names safely
feature_names = data.drop('diagnosis', axis=1).columns
if not hasattr(model, "feature_names_in_"):
    model.feature_names_in_ = feature_names

# -----------------------------
# STEP 3 — Streamlit Layout
# -----------------------------

# Sidebar: Project Info only
st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts whether a breast tumor is **Malignant** or **Benign** using a **Logistic Regression model**.

- Model trained on Breast Cancer dataset  
- Features include: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, etc.  
- Adjust input features below and get real-time predictions.
""")

# Main Page: Title
st.title("Breast Cancer Prediction")
st.subheader("Input Features")

# -----------------------------
# STEP 4 — Feature Inputs in Main Page
# -----------------------------
feature_inputs = {}
for feature in model.feature_names_in_:
    feature_inputs[feature] = st.number_input(
        label=feature,
        min_value=0.0,
        max_value=1000.0,
        value=float(data[feature].mean()),
        step=0.01,
        format="%.2f"
    )

# Create DataFrame and scale
input_df = pd.DataFrame(feature_inputs, index=[0])
input_scaled = scaler.transform(input_df)

# -----------------------------
# STEP 5 — Prediction
# -----------------------------
pred_class = model.predict(input_scaled)[0]
pred_prob = model.predict_proba(input_scaled)[0, 1]

st.subheader("Prediction Result")
if pred_class == 1:
    st.error(f"Prediction: Malignant (Probability: {pred_prob:.2f})")
else:
    st.success(f"Prediction: Benign (Probability: {1 - pred_prob:.2f})")

# -----------------------------
# STEP 6 — Evaluation Visuals (Optional)
# -----------------------------
X_full = data.drop('diagnosis', axis=1)
y_full = data['diagnosis']
X_scaled_full = scaler.transform(X_full)
y_pred_full = model.predict(X_scaled_full)

# ROC Curve
if st.checkbox("Show ROC Curve"):
    y_prob_full = model.predict_proba(X_scaled_full)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_full, y_prob_full)
    auc_score = roc_auc_score(y_full, y_prob_full)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt)

# Confusion Matrix
if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_full, y_pred_full)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    st.text("Classification Report:")
    report = classification_report(y_full, y_pred_full, target_names=['Benign', 'Malignant'])
    st.text(report)

# Feature Distribution Plots
if st.checkbox("Show Feature Distributions"):
    numeric_features = X_full.columns[:5]  # First 5 features
    for feature in numeric_features:
        plt.figure()
        sns.histplot(
            data,
            x=feature,
            hue='diagnosis',
            kde=True,
            palette={0: 'green', 1: 'red'},
            alpha=0.5
        )
        plt.title(f'Distribution of {feature} by Diagnosis')
        st.pyplot(plt)