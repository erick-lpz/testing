import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Predicción de la Severidad del Cáncer", layout="centered")
st.title("Predicción de Severidad del Cáncer")

MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo_severidad_cancer.pkl"

model_choice = st.radio("¿Cómo deseas cargar el modelo?", ("Desde MLflow", "Desde GitHub", "Subir modelo manualmente"))
model = None

# Cargar modelo según fuente
if model_choice == "Desde MLflow":
    mlflow_uri = st.text_input("URL del servidor MLflow (ngrok)").strip()
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            experiments = client.search_experiments()
            if experiments:
                exp_names = [e.name for e in experiments]
                selected_exp = st.selectbox("Selecciona un experimento", exp_names)
                exp_id = next(e.experiment_id for e in experiments if e.name == selected_exp)
                runs = client.search_runs([exp_id], order_by=["attributes.start_time desc"])
                run_ids = [r.info.run_id for r in runs]
                if run_ids:
                    selected_run = st.selectbox("Selecciona un modelo (run ID)", run_ids)
                    model_uri = f"runs:/{selected_run}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    st.success(f"Modelo MLflow cargado: {selected_run}")
                else:
                    st.warning("No hay modelos en ese experimento.")
            else:
                st.warning("No se encontraron experimentos.")
        except Exception as e:
            st.error(f"Error al conectar con MLflow: {e}")

elif model_choice == "Desde GitHub":
    try:
        st.info("Descargando modelo desde GitHub...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            model = joblib.load(BytesIO(response.content))
            st.success("Modelo cargado correctamente desde GitHub.")
        else:
            st.error(f"Fallo la descarga del modelo. Código: {response.status_code}")
    except Exception as e:
        st.error(f"Error al cargar modelo desde GitHub: {e}")

elif model_choice == "Subir modelo manualmente":
    uploaded_file = st.file_uploader("Sube tu modelo (.pkl)", type="pkl")
    if uploaded_file:
        try:
            model = joblib.load(uploaded_file)
            st.success("Modelo cargado desde archivo.")
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")

# Interfaz de usuario
if model is not None:
    st.subheader("Ingresar datos del paciente")

    age = st.number_input("Edad del paciente", 18, 100, 50)
    year = st.number_input("Año del diagnóstico", 2015, 2024, 2023)
    genetic_risk = st.slider("Riesgo genético", 0.0, 1.0, 0.5)
    air_pollution = st.slider("Contaminación del aire", 0.0, 100.0, 50.0)
    alcohol_use = st.slider("Consumo de alcohol", 0.0, 100.0, 20.0)
    smoking = st.slider("Tabaquismo", 0.0, 100.0, 30.0)
    obesity_level = st.slider("Obesidad", 0.0, 100.0, 25.0)
    treatment_cost = st.number_input("Costo del tratamiento (USD)", 0.0, 1_000_000.0, 20000.0)
    survival_years = st.slider("Años de supervivencia esperados", 0, 20, 5)

    gender = st.selectbox("Género", ["Male", "Female", "Other"])
    country_region = st.selectbox("País", ["USA", "UK", "India", "Russia", "China", "Brazil", "Pakistan", "Canada", "Germany"])
    cancer_type = st.selectbox("Tipo de cáncer", ["Lung", "Colon", "Skin", "Prostate", "Leukemia", "Cervical", "Liver"])
    cancer_stage = st.selectbox("Etapa del cáncer", ["Stage I", "Stage II", "Stage III", "Stage IV"])

    input_data = pd.DataFrame([{
        "Age": age,
        "Year": year,
        "Genetic_Risk": genetic_risk,
        "Air_Pollution": air_pollution,
        "Alcohol_Use": alcohol_use,
        "Smoking": smoking,
        "Obesity_Level": obesity_level,
        "Treatment_Cost_USD": treatment_cost,
        "Survival_Years": survival_years,
        "Gender": gender,
        "Country_Region": country,
        "Cancer_Type": cancer_type,
        "Cancer_Stage": cancer_stage
    }])

    try:
        input_encoded = pd.get_dummies(input_data, drop_first=True)

        # Alinear columnas
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[expected_cols]

        if st.button("Predecir severidad"):
            pred = model.predict(input_encoded)
            st.success(f"Severidad estimada: {pred[0]}")
    except Exception as e:
        st.error(f"Error en la codificación o predicción: {e}")

else:
    st.warning("Carga un modelo antes de predecir.")
