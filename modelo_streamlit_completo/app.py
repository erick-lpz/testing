import streamlit as st
import pandas as pd
import os
import requests

st.title("Predicción de Severidad de Cáncer")

# --- MLflow: Interfaz para seleccionar modelo (experimento/run) ---
model = None
mlflow_loaded = False
use_mlflow = st.checkbox("¿Cargar modelo desde MLflow?", value=False)

if use_mlflow:
    try:
        import mlflow
        import mlflow.pycaret
        from mlflow.tracking import MlflowClient

        # Cambia la URI por la de tu servidor MLflow/ngrok si es necesario
        MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiments = client.list_experiments()
        if experiments:
            experiment_names = [exp.name for exp in experiments]
            selected_exp = st.selectbox("Selecciona un experimento", experiment_names)
            exp_id = [exp.experiment_id for exp in experiments if exp.name == selected_exp][0]
            runs = client.search_runs(experiment_ids=[exp_id], order_by=["attributes.start_time desc"])
            run_ids = [run.info.run_id for run in runs]
            if run_ids:
                selected_run = st.selectbox("Selecciona un modelo (run)", run_ids)
                model_uri = f"runs:/{selected_run}/best_model"  # Cambia "best_model" si usaste otro nombre en log_model
                model = mlflow.pycaret.load_model(model_uri)
                mlflow_loaded = True
                st.success(f"Modelo MLflow cargado: {selected_run}")
            else:
                st.error("No hay modelos registrados para este experimento.")
        else:
            st.error("No hay experimentos en el servidor MLflow.")
    except Exception as e:
        st.warning(f"No se pudo conectar o cargar modelo desde MLflow: {e}")
        st.info("Puedes desactivar la opción de MLflow y usar el modelo local.")

# --- Si MLflow no está disponible, usa modelo.pkl local (descargándolo si hace falta) ---
if model is None:
    pkl_path = "modelo_streamlit_completo/modelo.pkl"
    modelo_url = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo.pkl"
    if not os.path.exists(pkl_path):
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        r = requests.get(modelo_url)
        if r.status_code == 200:
            with open(pkl_path, "wb") as f:
                f.write(r.content)
            st.info("Modelo local descargado dinámicamente desde GitHub.")
        else:
            st.error(f"No se pudo descargar el modelo local ({r.status_code})")
    try:
        from pycaret.classification import load_model, predict_model
        model = load_model(pkl_path)
        st.success("Modelo local cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo local: {e}")

# --- Interfaz de usuario y predicción ---
if model is not None:
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
            from pycaret.classification import predict_model
            result = predict_model(model, data=df)
            pred_label = result['prediction_label'][0] if 'prediction_label' in result.columns else result['Label'][0]
            st.success(f"La predicción de severidad es: {pred_label}")
        except Exception as e:
            try:
                pred = model.predict(df)
                st.success(f"La predicción de severidad es: {pred[0]}")
            except Exception as e2:
                st.error(f"Ocurrió un error al predecir: {e2}")
else:
    st.error("No fue posible cargar ningún modelo para las predicciones.")
