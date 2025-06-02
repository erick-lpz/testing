import streamlit as st
import pandas as pd
import os
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Predicción de Severidad de Cáncer", layout="centered")
st.title("🔬 Predicción de Severidad de Cáncer")

# --- CONFIG: Ruta del modelo en GitHub ---
MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo_severidad_cancer.pkl"

# --- Selección del origen del modelo ---
model_choice = st.radio(
    "¿Cómo deseas cargar el modelo?",
    ("Desde MLflow", "Desde GitHub", "Subir modelo manualmente")
)

model = None
mlflow_error = None

# --- Opción 1: MLflow ---
if model_choice == "Desde MLflow":
    mlflow_uri = st.text_input(
        "URL del servidor MLflow (ngrok)",
        value=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="Ejemplo: https://abcd1234.ngrok-free.app"
    ).strip()

    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            experiments = client.search_experiments()
            if experiments:
                experiment_names = [exp.name for exp in experiments]
                selected_exp = st.selectbox("Selecciona un experimento", experiment_names)
                exp_id = [exp.experiment_id for exp in experiments if exp.name == selected_exp][0]
                runs = client.search_runs([exp_id], order_by=["attributes.start_time desc"])
                run_ids = [run.info.run_id for run in runs]

                if run_ids:
                    selected_run = st.selectbox("Selecciona un modelo (run ID)", run_ids)
                    model_uri = f"runs:/{selected_run}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    st.success(f"Modelo MLflow cargado: {selected_run}")
                else:
                    mlflow_error = "⚠️ No hay modelos registrados en ese experimento."
            else:
                mlflow_error = "⚠️ No se encontraron experimentos en MLflow."
        except Exception as e:
            mlflow_error = f"❌ Error al conectar con MLflow: {e}"

# --- Opción 2: GitHub ---
if model_choice == "Desde GitHub":
    try:
        st.info("Descargando modelo desde GitHub...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            model = joblib.load(BytesIO(response.content))
            st.success("✅ Modelo cargado correctamente desde GitHub.")
        else:
            st.error(f"❌ Falló la descarga del modelo. Código: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error al cargar modelo desde GitHub: {e}")

# --- Opción 3: Subida manual ---
if model_choice == "Subir modelo manualmente":
    uploaded_file = st.file_uploader("Sube tu modelo (.pkl)", type="pkl")
    if uploaded_file:
        try:
            model = joblib.load(uploaded_file)
            st.success("✅ Modelo cargado desde archivo.")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")

# --- Interfaz de predicción si hay modelo ---
if model is not None:
    st.subheader("📋 Ingresar datos del paciente")

    # Entradas del usuario (solo las 9 columnas necesarias)
    age = st.number_input("Edad del paciente", min_value=18, max_value=100, value=50)
    year = st.number_input("Año del diagnóstico", min_value=2015, max_value=2024, value=2023)
    genetic_risk = st.slider("Riesgo genético (0 a 1)", 0.0, 1.0, 0.5)
    air_pollution = st.slider("Contaminación del aire", 0.0, 100.0, 50.0)
    alcohol_use = st.slider("Consumo de alcohol", 0.0, 100.0, 20.0)
    smoking = st.slider("Tabaquismo", 0.0, 100.0, 30.0)
    obesity_level = st.slider("Obesidad", 0.0, 100.0, 25.0)
    treatment_cost = st.number_input("Costo del tratamiento (USD)", 0.0, 1_000_000.0, 20000.0)
    survival_years = st.slider("Años de supervivencia esperados", 0, 20, 5)

    # Crear DataFrame de entrada
    input_df = pd.DataFrame([{
        'Age': age,
        'Year': year,
        'Genetic_Risk': genetic_risk,
        'Air_Pollution': air_pollution,
        'Alcohol_Use': alcohol_use,
        'Smoking': smoking,
        'Obesity_Level': obesity_level,
        'Treatment_Cost_USD': treatment_cost,
        'Survival_Years': survival_years
    }])

    # Botón de predicción
    if st.button("🔍 Predecir severidad"):
        try:
            pred = model.predict(input_df)
            st.success(f"🧬 Severidad estimada: {pred[0]}")
        except Exception as e:
            st.error(f"❌ Error durante la predicción: {e}")
else:
    st.warning("⚠️ Carga un modelo antes de predecir.")
