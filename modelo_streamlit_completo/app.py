import streamlit as st
import pandas as pd
import os
import joblib

st.title("Predicción de Severidad de Cáncer")

# --- CONFIG: Ruta del modelo local (siempre usa esta ruta para el .pkl) ---
MODEL_PATH = "modelo_streamlit_completo/modelo.pkl"

# --- INPUT: URL de MLflow (desde la interfaz, no hardcodeado) ---
mlflow_uri = st.text_input(
    "URL de MLflow (ngrok, opcional)",
    value=os.environ.get("MLFLOW_TRACKING_URI", ""),
    help="Pega aquí la URL de tu servidor MLflow de Colab/ngrok, ejemplo: https://1234.ngrok-free.app"
).strip()

model = None
mlflow_error = None

# --- PRIMER INTENTO: Cargar modelo desde MLflow si se ingresa URL ---
if mlflow_uri:
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        from mlflow.tracking import MlflowClient
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
                model_uri = f"runs:/{selected_run}/model"
                model = mlflow.sklearn.load_model(model_uri)
                st.success(f"Modelo MLflow cargado: {selected_run}")
            else:
                mlflow_error = "No hay modelos registrados para este experimento."
        else:
            mlflow_error = "No hay experimentos en el servidor MLflow."
    except Exception as e:
        mlflow_error = f"No se pudo conectar o cargar modelo desde MLflow: {e}"

# --- FALLBACK: Modelo local (si MLflow no se usó, o falló) ---
if model is None:
    if mlflow_uri and mlflow_error:
        st.warning(f"{mlflow_error}\n\nPuedes dejar vacío el campo de MLflow y usar el modelo local.")
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.success("Modelo local cargado correctamente.")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo local: {e}")
    else:
        st.error(f"No se encontró el modelo local en {MODEL_PATH}. Sube el archivo modelo.pkl a esa ruta.")

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
            pred = model.predict(df)
            st.success(f"La predicción de severidad es: {pred[0]}")
        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
else:
    st.warning("No fue posible cargar ningún modelo para las predicciones.")
