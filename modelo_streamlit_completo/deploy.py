import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Predicción de Severidad de Cáncer", layout="centered")
st.title("Predicción de la Severidad del Cáncer")

MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo_severidad_cancer_svs%20(1).pkl"
severidad_map = {0: "Baja", 1: "Media", 2: "Alta"}

model = None
is_pycaret = False

opcion = st.radio("¿Cómo deseas cargar el modelo?", ("Desde MLflow", "Desde GitHub", "Subir manualmente"))

if opcion == "Desde MLflow":
    uri = st.text_input("URL del servidor MLflow (ngrok)").strip()
    if uri:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(uri)
            client = MlflowClient()
            exps = client.search_experiments()
            if exps:
                exp = st.selectbox("Selecciona un experimento", [e.name for e in exps])
                exp_id = next(e.experiment_id for e in exps if e.name == exp)
                runs = client.search_runs([exp_id], order_by=["start_time desc"])
                run_ids = [r.info.run_id for r in runs]
                if run_ids:
                    run = st.selectbox("Selecciona un modelo (run ID)", run_ids)
                    model = mlflow.pyfunc.load_model(f"runs:/{run}/model")
                    st.success(f"Modelo cargado: {run}")
                else:
                    st.warning("No hay modelos en ese experimento.")
            else:
                st.warning("No se encontraron experimentos.")
        except Exception as e:
            st.error(f"Error MLflow: {e}")

elif opcion == "Desde GitHub":
    try:
        st.info("Descargando modelo desde GitHub...")
        r = requests.get(MODEL_URL)
        with open("model.pkl", "wb") as f:
            f.write(r.content)
        from pycaret.classification import load_model
        model = load_model("model")
        is_pycaret = True
        os.remove("model.pkl")
        if os.path.exists("model"): os.remove("model")
        st.success("Modelo cargado desde GitHub.")
    except Exception as e:
        st.error(f"Error: {e}")

elif opcion == "Subir manualmente":
    file = st.file_uploader("Sube tu modelo `.pkl`", type="pkl")
    if file:
        try:
            with open("user_model.pkl", "wb") as f:
                f.write(file.getbuffer())
            from pycaret.classification import load_model
            model = load_model("user_model")
            is_pycaret = True
            os.remove("user_model.pkl")
            if os.path.exists("user_model"): os.remove("user_model")
            st.success("Modelo cargado correctamente.")
        except Exception as e:
            st.error(f"Error: {e}")

if model:
    st.subheader("Datos del paciente")
    datos = {
        'Genetic_Risk': st.number_input("Riesgo genético", 0.0, 10.0, 5.0),
        'Air_Pollution': st.number_input("Contaminación del aire", 0.0, 10.0, 5.0),
        'Alcohol_Use': st.number_input("Consumo de alcohol", 0.0, 10.0, 5.0),
        'Smoking': st.number_input("Tabaquismo", 0.0, 10.0, 5.0),
        'Obesity_Level': st.number_input("Obesidad", 0.0, 10.0, 5.0),
        'Treatment_Cost_USD': st.number_input("Costo del tratamiento (USD)", 0.0, 100000.0, 50000.0),
        'Survival_Years': st.number_input("Años de supervivencia", 0.0, 15.0, 5.0),
        'Cancer_Stage': st.radio("Etapa del cáncer", ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"])
    }

    tipo = st.selectbox("Tipo de cáncer", ["Cervical", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin"])
    for t in ["Cervical", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin"]:
        datos[f'Cancer_Type_{t}'] = (t == tipo)

    if st.button("Predecir severidad"):
        try:
            df = pd.DataFrame([datos])
            if is_pycaret:
                from pycaret.classification import predict_model
                pred = predict_model(model, data=df)['prediction_label'].iloc[0]
            else:
                pred = model.predict(df)[0]
            st.success(f"Severidad estimada: **{severidad_map.get(pred, 'Desconocida')}**")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
