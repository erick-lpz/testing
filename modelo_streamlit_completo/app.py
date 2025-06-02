import streamlit as st
import pandas as pd
import os

st.title("Predicción de Severidad de Cáncer")

model = None

# 1. Intentar cargar con MLflow primero
try:
    import mlflow.pycaret
    if os.path.exists("modelo_streamlit_completo/modelo.pkl"):
        model = mlflow.pycaret.load_model("modelo_streamlit_completo/modelo.pkl")
        st.info("Modelo cargado desde modelo.pkl usando MLflow.")
except Exception as e:
    st.warning(f"No se pudo cargar modelo.pkl con MLflow: {e}")

# 2. Si no se pudo, intenta con PyCaret
if model is None:
    try:
        from pycaret.classification import load_model, predict_model
        if os.path.exists("modelo_streamlit_completo/modelo.pgl"):
            model = load_model("modelo_streamlit_completo/modelo.pgl")
            st.info("Modelo cargado desde modelo.pgl usando PyCaret.")
        else:
            st.error("No se encontró modelo.pgl en el directorio.")
    except Exception as e:
        st.error(f"No se pudo cargar modelo.pgl con PyCaret: {e}")

if model is not None:
    # Entradas del usuario
    year = st.number_input("Año del diagnóstico", min_value=2015, max_value=2024)
    genetic_risk = st.slider("Riesgo genético (0 a 1)", 0.0, 1.0, 0.5)
    air_pollution = st.slider("Contaminación del aire", 0.0, 100.0, 50.0)
    alcohol_use = st.slider("Consumo de alcohol", 0.0, 100.0, 20.0)
    smoking = st.slider("Tabaquismo", 0.0, 100.0, 30.0)
    obesity_level = st.slider("Obesidad", 0.0, 100.0, 25.0)
    treatment_cost = st.number_input("Costo del tratamiento (USD)", 0.0, 1000000.0, 20000.0)
    survival_years = st.slider("Años de supervivencia esperados", 0, 20, 5)
    gender = st.selectbox("Género", ["Male", "Female", "Other"])
    country = st.selectbox("País", ["USA", "UK", "India", "Russia", "China", "Brazil", "Pakistan", "Canada", "Germany"])
    cancer_type = st.selectbox("Tipo de cáncer", ["Lung", "Colon", "Skin", "Prostate", "Leukemia", "Cervical", "Liver"])
    cancer_stage = st.selectbox("Etapa del cáncer", ["Stage I", "Stage II", "Stage III", "Stage IV"])

    if st.button("Predecir severidad"):
        df = pd.DataFrame({
            'Year': [year],
            'Genetic_Risk': [genetic_risk],
            'Air_Pollution': [air_pollution],
            'Alcohol_Use': [alcohol_use],
            'Smoking': [smoking],
            'Obesity_Level': [obesity_level],
            'Treatment_Cost_USD': [treatment_cost],
            'Survival_Years': [survival_years],
            'Gender': [gender],
            'Country_Region': [country],
            'Cancer_Type': [cancer_type],
            'Cancer_Stage': [cancer_stage]
        })

        try:
            if "predict_model" in globals():
                result = predict_model(model, data=df)
                pred_label = result['prediction_label'][0] if 'prediction_label' in result.columns else result['Label'][0]
                st.success(f"La predicción de severidad es: {pred_label}")
            else:
                # Si es MLflow, el modelo sigue la interfaz sklearn
                pred = model.predict(df)
                st.success(f"La predicción de severidad es: {pred[0]}")
        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
else:
    st.error("No fue posible cargar ningún modelo para las predicciones.")
