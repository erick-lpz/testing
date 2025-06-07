import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Predicción de Severidad de Cáncer", layout="centered")
st.title("Predicción de la Severidad del Cáncer")

MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo_severidad_cancer_svs.pkl"

model_choice = st.radio("¿Cómo deseas cargar el modelo?", ("Desde MLflow", "Desde GitHub", "Subir modelo manualmente"))
model = None
column_order = None
is_pycaret = False

# Cargar el modelo
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
                    model = mlflow.pyfunc.load_model(model_uri)
                    st.success(f"Modelo MLflow cargado: {selected_run}")
                else:
                    st.warning("No hay modelos disponibles en ese experimento.")
            else:
                st.warning("No se encontraron experimentos.")
        except Exception as e:
            st.error(f"Error al conectar con MLflow: {e}")

elif model_choice == "Desde GitHub":
    try:
        st.info("Descargando modelo desde GitHub...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            # Guardar temporalmente para cargar con PyCaret
            with open("temp_model.pkl", "wb") as f:
                f.write(response.content)
            try:
                from pycaret.classification import load_model
                model = load_model("temp_model")
                is_pycaret = True
                st.success("Modelo PyCaret cargado correctamente desde GitHub.")
            except Exception as e:
                st.error(f"Error al cargar el modelo PyCaret: {e}")
            finally:
                # Elimina el archivo temporal después de cargar
                if os.path.exists("temp_model.pkl"):
                    os.remove("temp_model.pkl")
                if os.path.exists("temp_model"):
                    os.remove("temp_model")
        else:
            st.error(f"Error al descargar el modelo. Código de estado: {response.status_code}")
    except Exception as e:
        st.error(f"Error al cargar modelo desde GitHub: {e}")

elif model_choice == "Subir modelo manualmente":
    uploaded_file = st.file_uploader("Sube tu modelo `.pkl` de PyCaret", type="pkl")
    if uploaded_file:
        try:
            with open("uploaded_model.pkl", "wb") as f:
                f.write(uploaded_file.getbuffer())
            from pycaret.classification import load_model
            model = load_model("uploaded_model")
            is_pycaret = True
            st.success("Modelo PyCaret cargado exitosamente desde archivo.")
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
        finally:
            if os.path.exists("uploaded_model.pkl"):
                os.remove("uploaded_model.pkl")
            if os.path.exists("uploaded_model"):
                os.remove("uploaded_model")

# Formulario de entrada de datos
if model:
    st.subheader("Ingresar datos del paciente")

    input_data = {}

    # Variables numéricas
    input_data['Genetic_Risk'] = st.number_input("Riesgo genético", 0.0, 10.0, 5.0)
    input_data['Air_Pollution'] = st.number_input("Contaminación del aire", 0.0, 10.0, 5.0)
    input_data['Alcohol_Use'] = st.number_input("Consumo de alcohol", 0.0, 10.0, 5.0)
    input_data['Smoking'] = st.number_input("Tabaquismo", 0.0, 10.0, 5.0)
    input_data['Obesity_Level'] = st.number_input("Obesidad", 0.0, 10.0, 5.0)
    input_data['Treatment_Cost_USD'] = st.number_input("Costo del tratamiento (USD)", 0.0, 100000.0, 50000.0)
    input_data['Survival_Years'] = st.number_input("Años de supervivencia", 0.0, 15.0, 5.0)

    # Etapa del cáncer
    stage = st.radio("Etapa del cáncer", ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"])
    input_data['Cancer_Stage'] = stage

    # Tipo de cáncer (one-hot)
    tipos = ["Cervical", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin"]
    tipo_seleccionado = st.selectbox("Tipo de cáncer", tipos)
    for tipo in tipos:
        input_data[f'Cancer_Type_{tipo}'] = (tipo == tipo_seleccionado)

    if st.button("Predecir severidad"):
        try:
            df_input = pd.DataFrame([input_data])
            if is_pycaret:
                from pycaret.classification import predict_model
                resultado = predict_model(model, data=df_input)
                prediccion = resultado['prediction_label'].iloc[0]
                st.success(f"Severidad estimada: **{prediccion}**")
            else:
                # Para modelos MLflow
                prediction = model.predict(df_input)
                if hasattr(prediction, '__iter__'):
                    prediccion = prediction[0]
                else:
                    prediccion = prediction
                st.success(f"Severidad estimada: **{prediccion}**")
        except Exception as e:
            st.error(f"Error al hacer la predicción: {e}")
