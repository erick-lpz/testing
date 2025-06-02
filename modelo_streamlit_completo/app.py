import streamlit as st
import pandas as pd
import os
import joblib
import requests
from io import BytesIO

st.title("Predicción de Severidad de Cáncer")

# --- CONFIG: Ruta del modelo local ---
MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo.pkl"  # El enlace directo al modelo en GitHub

# --- INTERFAZ DE USUARIO: Selección del tipo de modelo ---
model_choice = st.radio(
    "¿Cómo deseas cargar el modelo?",
    ("Desde MLflow", "Desde archivo local", "Subir modelo manualmente")
)

model = None
mlflow_error = None

# --- Cargar modelo desde MLflow ---
if model_choice == "Desde MLflow":
    mlflow_uri = st.text_input(
        "URL de MLflow (ngrok, opcional)",
        value=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="Pega aquí la URL de tu servidor MLflow de Colab/ngrok, ejemplo: https://1234.ngrok-free.app"
    ).strip()

    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)  # Establecer URI de MLflow
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            # Intentar obtener los experimentos (usando search_experiments en vez de list_experiments)
            experiments = client.search_experiments()  # Obtener los experimentos

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

# --- Cargar modelo desde archivo local desde GitHub ---
if model_choice == "Desde archivo local":
    try:
        st.info("Descargando modelo desde GitHub...")
        # Descargar el modelo desde GitHub
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            model = joblib.load(BytesIO(response.content))  # Cargar el modelo directamente desde el contenido descargado
            st.success("Modelo local cargado correctamente desde GitHub.")
        else:
            st.error(f"No se pudo descargar el modelo desde GitHub. Código de estado: {response.status_code}")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo local desde GitHub: {e}")

# --- Subir un archivo de modelo manualmente ---
if model_choice == "Subir modelo manualmente":
    uploaded_file = st.file_uploader("Sube tu modelo (.pkl)", type="pkl")
    if uploaded_file:
        try:
            model = joblib.load(uploaded_file)
            st.success("Modelo cargado desde el archivo.")
        except Exception as e:
            st.error(f"Error al cargar el modelo desde el archivo: {e}")

# --- Interfaz de usuario y predicción ---
if model is not None:
    # **Mantener el año como estaba antes**
    year = st.number_input("Año del diagnóstico", min_value=2015, max_value=2024)

    # Asegúrate de que todas las columnas que el modelo espera estén presentes
    age = st.number_input("Edad del paciente", min_value=18, max_value=100, value=50)

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

    # Crear DataFrame asegurándonos de incluir todas las columnas necesarias
df = pd.DataFrame({
    'Age': [age],  # Asegúrate de agregar 'Age'
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

# Transformar las columnas categóricas a one-hot encoding (variables dummy)
df_encoded = pd.get_dummies(df, drop_first=True)

# Verificar si el DataFrame contiene todas las columnas necesarias
model_columns = model.feature_names_in_  # Obtener las columnas que el modelo espera
missing_cols = [col for col in model_columns if col not in df_encoded.columns]

if missing_cols:
    st.error(f"Faltan las siguientes columnas: {', '.join(missing_cols)}")
else:
    # Si todo está correcto, hacer la predicción
    if st.button("Predecir severidad"):
        try:
            pred = model.predict(df_encoded)
            st.success(f"La predicción de severidad es: {pred[0]}")
        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
