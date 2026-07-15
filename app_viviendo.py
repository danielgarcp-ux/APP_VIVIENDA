import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Valuador Inmobiliario AI", layout="wide")

# --- 1. CARGA DE DATOS Y ENTRENAMIENTO EN CACHÉ ---
@st.cache_resource
def preparar_modelo():
    # Cargar el dataset que subiste a GitHub
    df = pd.read_csv('dataset_inmobiliario_lima.csv')
    
    # Extraer listas únicas para los selectores de la interfaz
    listas_ui = {
        'distritos': sorted(df['Distrito'].unique()),
        'zonificacion': sorted(df['Zonificacion'].unique()),
        'modalidad': sorted(df['Modalidad_Compra'].unique())
    }
    
    # Preprocesamiento (One-Hot Encoding)
    columnas_categoricas = ['Distrito', 'Zonificacion', 'Modalidad_Compra']
    df_procesado = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
    
    X = df_procesado.drop('Precio_Publicado_USD', axis=1)
    y = df_procesado['Precio_Publicado_USD']
    
    # Entrenamiento del XGBoost con todos los datos
    modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    
    return modelo, X.columns, listas_ui

modelo_xgb, columnas_modelo, opciones_ui = preparar_modelo()

# --- 2. INTERFAZ DE USUARIO ---
st.title("🏡 Valuador Inmobiliario con IA (XGBoost)")
st.markdown("Ingresa los datos del inmueble para obtener un dictamen financiero de mercado.")

with st.form("formulario_inmueble"):
    st.subheader("Datos de Ubicación e Infraestructura")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distrito = st.selectbox("Distrito", opciones_ui['distritos'])
        area_total = st.number_input("Área Total (m²)", min_value=10, max_value=10000, value=120)
        area_techada = st.number_input("Área Techada (m²)", min_value=0, max_value=10000, value=0)
        
    with col2:
        habitaciones = st.number_input("Habitaciones", min_value=0, max_value=20, value=0)
        banos = st.number_input("Baños", min_value=0, max_value=10, value=0)
        estacionamientos = st.number_input("Estacionamientos", min_value=0, max_value=10, value=0)
        
    with col3:
        antiguedad = st.number_input("Antigüedad (Años)", min_value=0, max_value=100, value=0)
        latitud = st.number_input("Latitud (Aprox)", value=-12.04)
        longitud = st.number_input("Longitud (Aprox)", value=-77.02)

    st.subheader("Datos Técnicos y Comerciales")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        seguridad = st.slider("Índice de Seguridad (1-10)", 1, 10, 5)
        prox_vias = st.number_input("Proximidad Vías Principales (km)", min_value=0.0, value=1.0)
        zonificacion = st.selectbox("Zonificación", opciones_ui['zonificacion'])
        
    with col5:
        viabilidad_pozo = st.checkbox("Viabilidad para Pozo de Agua")
        potencial_solar = st.checkbox("Potencial Energía Solar")
        independizado = st.checkbox("Estado: Independizado en RRPP")
        
    with col6:
        modalidad = st.selectbox("Modalidad de Compra", opciones_ui['modalidad'])
        precio_publicado = st.number_input("Precio de Venta Solicitado (USD)", min_value=1000, value=50000, step=1000)

    submit = st.form_submit_button("Analizar Precio")

# --- 3. LÓGICA DE PREDICCIÓN Y DICTAMEN ---
if submit:
    # 3.1 Construir el diccionario con los datos ingresados
    datos_usuario = {
        'Latitud': latitud,
        'Longitud': longitud,
        'Indice_Seguridad': seguridad,
        'Proximidad_Vias_Principales_km': prox_vias,
        'Area_Total_m2': area_total,
        'Area_Techada_m2': area_techada,
        'Habitaciones': habitaciones,
        'Banos': banos,
        'Antiguedad_Anios': antiguedad,
        'Estacionamientos': estacionamientos,
        'Viabilidad_Pozo_Agua': 1 if viabilidad_pozo else 0,
        'Potencial_Energia_Solar': 1 if potencial_solar else 0,
        'Estado_Independizacion': 1 if independizado else 0,
        'Distrito': distrito,
        'Zonificacion': zonificacion,
        'Modalidad_Compra': modalidad
    }
    
    # 3.2 Convertir a DataFrame y aplicar One-Hot Encoding
    df_input = pd.DataFrame([datos_usuario])
    columnas_categoricas = ['Distrito', 'Zonificacion', 'Modalidad_Compra']
    df_input_procesado = pd.get_dummies(df_input, columns=columnas_categoricas)
    
    # 3.3 Alinear las columnas del input con las que espera el modelo entrenado
    df_final = df_input_procesado.reindex(columns=columnas_modelo, fill_value=0)
    
    # 3.4 Realizar Predicción
    precio_estimado = modelo_xgb.predict(df_final)[0]
    
    # 3.5 Dictamen
    diferencia_porcentual = (precio_publicado - precio_estimado) / precio_estimado
    
    st.divider()
    st.subheader("📊 Resultados de la Evaluación")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio Solicitado", f"USD {precio_publicado:,.2f}")
    c2.metric("Precio Estimado XGBoost", f"USD {precio_estimado:,.2f}")
    c3.metric("Variación", f"{diferencia_porcentual:.1%}")
    
    if diferencia_porcentual > 0.07:
        st.error(f"🔴 **CARO:** El precio está sobrevalorado respecto a la infraestructura y características de la zona.")
    elif diferencia_porcentual < -0.07:
        st.success(f"🟢 **BARATO:** El precio está por debajo del mercado. Representa una excelente oportunidad de inversión o arbitraje.")
    else:
        st.info(f"🔵 **NORMAL:** El precio es justo y se alinea correctamente con las tendencias del mercado local.")