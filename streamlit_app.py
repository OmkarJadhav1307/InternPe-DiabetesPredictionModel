# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Streamlit page settings
st.set_page_config(page_title="Diabetes Prediction (Logistic Regression)", layout="centered", page_icon="🧬", initial_sidebar_state="expanded")

# Title
st.title("🧬 Diabetes Prediction using Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload Diabetes CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing
    cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zero_as_missing] = data[cols_with_zero_as_missing].replace(0, np.nan)
    data[cols_with_zero_as_missing] = data[cols_with_zero_as_missing].fillna(data[cols_with_zero_as_missing].median())

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    st.success("Logistic Regression model trained successfully!")

    # Model evaluation
    y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.4).astype(int)  # THRESHOLD = 0.4

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.subheader("Model Evaluation")
    st.write(f"**Accuracy**: {acc:.4f}")
    st.write(f"**Precision**: {prec:.4f}")
    st.write(f"**Recall**: {rec:.4f}")
    st.write(f"**F1 Score**: {f1:.4f}")
    st.write(f"**ROC AUC**: {roc_auc:.4f}")

    # Sidebar for user input
    st.sidebar.header("Patient Data Input")

    def user_input_features():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 2)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        blood_pressure = st.sidebar.slider('Blood Pressure', 0, 140, 70)
        skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0.0, 900.0, 100.0)
        bmi = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
        age = st.sidebar.slider('Age', 10, 100, 30)

        user_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        return pd.DataFrame(user_data, index=[0])

    input_df = user_input_features()

    # Predict button
    if st.sidebar.button("Predict Diabetes"):
        input_scaled = scaler.transform(input_df)
        prediction_proba = log_reg.predict_proba(input_scaled)[:, 1]
        prediction = (prediction_proba >= 0.4).astype(int)

        if prediction[0] == 1:
            st.error(f"🔴 High risk of Diabetes! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.success(f"🟢 Low risk of Diabetes (Probability: {1 - prediction_proba[0]:.2f})")
