import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Cargar modelo
model = load_model("modelo_severidad_cancer")

st.title("🧬 Predicción de Severidad del Cáncer")
st.write("Completa el formulario con los datos del paciente para predecir la severidad.")

with st.form("formulario"):
    age = st.slider("Edad", 0, 100, 50)
    gender = st.selectbox("Género", ["Male", "Female"])
    cancer_type = st.selectbox("Tipo de Cáncer", ["Lung", "Breast", "Colon", "Other"])
    smoking = st.selectbox("Historial de Tabaquismo", ["Yes", "No"])
    
    enviar = st.form_submit_button("Predecir")

    if enviar:
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Cancer_Type": cancer_type,
            "Smoking_History": smoking
        }])
        prediction = predict_model(model, data=input_data)
        pred_label = prediction['prediction_label'][0]
        pred_score = prediction['prediction_score'][0]

        st.success(f"🧾 Predicción: **{pred_label}**")
        st.info(f"🔢 Score de confianza: **{round(pred_score*100, 2)}%**")