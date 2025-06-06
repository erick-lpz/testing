import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Predicción de Severidad de Cáncer", layout="centered")
st.title("Predicción de la Severidad del Cáncer")

MODEL_URL = "https://github.com/erick-lpz/testing/raw/main/modelo_streamlit_completo/modelo_severidad_cancer_svs.pkl"

model_choice = st.radio("¿Cómo deseas cargar el modelo?", ("Desde GitHub", "Subir modelo manualmente"))
model = None

# === Cargar el modelo ===
if model_choice == "Desde GitHub":
    try:
        st.info("Descargando modelo desde GitHub...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            # Guardar temporalmente para cargar con PyCaret
            with open("temp_model.pkl", "wb") as f:
                f.write(response.content)
            model = load_model("temp_model")  # Sin extensión .pkl
            st.success("Modelo PyCaret cargado correctamente desde GitHub.")
        else:
            st.error(f"Error al descargar el modelo. Código de estado: {response.status_code}")
    except Exception as e:
        st.error(f"Error al cargar modelo desde GitHub: {e}")

elif model_choice == "Subir modelo manualmente":
    uploaded_file = st.file_uploader("Sube tu modelo `.pkl` de PyCaret", type="pkl")
    if uploaded_file:
        try:
            # Guardar temporalmente
            with open("uploaded_model.pkl", "wb") as f:
                f.write(uploaded_file.getbuffer())
            model = load_model("uploaded_model")  # Sin extensión .pkl
            st.success("Modelo PyCaret cargado exitosamente desde archivo.")
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")

# === Formulario de entrada de datos ===
if model:
    st.subheader("Ingresar datos del paciente")

    input_data = {}

    # Variables numéricas
    input_data['Genetic_Risk'] = st.number_input("Riesgo Genético", 0.0, 10.0, 5.0)
    input_data['Air_Pollution'] = st.number_input("Contaminación del Aire", 0.0, 10.0, 5.0)
    input_data['Alcohol_Use'] = st.number_input("Consumo de Alcohol", 0.0, 10.0, 5.0)
    input_data['Smoking'] = st.number_input("Tabaquismo", 0.0, 10.0, 5.0)
    input_data['Obesity_Level'] = st.number_input("Obesidad", 0.0, 10.0, 5.0)
    input_data['Treatment_Cost_USD'] = st.number_input("Costo del Tratamiento (USD)", 0.0, 100000.0, 50000.0)
    input_data['Survival_Years'] = st.number_input("Años de Supervivencia", 0.0, 15.0, 5.0)

    # Etapa del cáncer
    stage = st.radio("Etapa del Cáncer", ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"])
    input_data['Cancer_Stage'] = stage

    # Tipo de cáncer (one-hot manual)
    tipos = ["Cervical", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin"]
    tipo_seleccionado = st.selectbox("Tipo de Cáncer", tipos)
    for tipo in tipos:
        input_data[f'Cancer_Type_{tipo}'] = (tipo == tipo_seleccionado)

    if st.button("Predecir Severidad"):
        try:
            df_input = pd.DataFrame([input_data])
            resultado = predict_model(model, data=df_input)
            prediccion = resultado['prediction_label'].iloc[0]
            st.success(f"Severidad estimada: **{prediccion}**")
        except Exception as e:
            st.error(f"Error al hacer la predicción: {e}")
