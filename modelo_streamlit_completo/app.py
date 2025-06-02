
import streamlit as st
import pandas as pd
import pickle

st.title("Predicción de Severidad de Cáncer")

model = mlflow.pycaret.load_model("modelo_severidad_cancer")

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
    model = pickle.load(open("modelo.pkl", "rb"))

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

    df_encoded = pd.get_dummies(df)
    columnas_esperadas = model.feature_names_in_
    for col in columnas_esperadas:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columnas_esperadas]

    prediction = model.predict(df_encoded)[0]
    st.success(f"La predicción de severidad es: {prediction}")
