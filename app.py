import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# 1. Configuración de la página (Tema Abierto / Claro)
st.set_page_config(
    page_title="Smart City: Analítica de Calidad del Aire",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para mantener el tema abierto y limpio
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stMetric { background-color: #f4f6f9; padding: 15px; border-radius: 10px; border-left: 5px solid #2E86AB; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Monitor de Analítica Avanzada: Calidad del Aire")
st.markdown("Este dashboard utiliza **Machine Learning** para predecir los niveles de contaminación (PM2.5) basándose en variables ambientales y de tráfico urbano.")
st.markdown("---")

# 2. Generación de datos ambientales simulados (Smart City)
@st.cache_data
def load_data():
    np.random.seed(42)
    fechas = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    
    data = []
    for fecha in fechas:
        # Simulamos estacionalidad: más tráfico en invierno/otoño
        mes = fecha.month
        factor_estacion = 1.5 if mes in [11, 12, 1, 2] else 1.0
        
        # Variables independientes
        trafico = np.random.normal(50000, 10000) * factor_estacion # Vehículos diarios
        viento = np.random.uniform(2, 25) # km/h
        temperatura = np.random.normal(18, 8) # °C
        
        # Fórmula base para PM2.5 (µg/m³): Más tráfico = más contaminación. Más viento = menos contaminación.
        pm25 = (trafico * 0.0008) - (viento * 0.5) + np.random.normal(0, 5)
        pm25 = max(5, pm25) # Evitar valores negativos
        
        data.append([fecha, trafico, viento, temperatura, pm25])
        
    df = pd.DataFrame(data, columns=['Fecha', 'Tráfico (Vehículos)', 'Viento (km/h)', 'Temperatura (°C)', 'PM2.5 (µg/m³)'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

df_aire = load_data()

# 3. Barra lateral con filtros
st.sidebar.header("⚙️ Filtros Temporales")
rango_fechas = st.sidebar.date_input(
    "Selecciona el rango de fechas:",
    [df_aire['Fecha'].min(), df_aire['Fecha'].max()]
)

# Filtrar por fecha
if len(rango_fechas) == 2:
    start_date, end_date = rango_fechas
    df_filtered = df_aire[(df_aire['Fecha'].dt.date >= start_date) & (df_aire['Fecha'].dt.date <= end_date)]
else:
    df_filtered = df_aire

# 4. KPIs de Sostenibilidad
st.subheader("📊 Resumen Ejecutivo Ambiental")
col1, col2, col3, col4 = st.columns(4)

pm25_promedio = df_filtered['PM2.5 (µg/m³)'].mean()
dias_criticos = len(df_filtered[df_filtered['PM2.5 (µg/m³)'] > 15]) # Límite OMS
trafico_promedio = df_filtered['Tráfico (Vehículos)'].mean()
viento_promedio = df_filtered['Viento (km/h)'].mean()

with col1:
    st.metric("PM2.5 Promedio", f"{pm25_promedio:.1f} µg/m³")
with col2:
    st.metric("Días Críticos (OMS)", f"{dias_criticos} días", delta="Alerta" if dias_criticos > 50 else "Normal", delta_color="inverse")
with col3:
    st.metric("Tráfico Promedio", f"{trafico_promedio:,.0f} veh/día")
with col4:
    st.metric("Viento Promedio", f"{viento_promedio:.1f} km/h")

st.markdown("---")

# 5. Pestañas de Análisis y Predicción
tab1, tab2, tab3 = st.tabs(["📈 Evolución Histórica", "🔍 Correlación de Variables", "🤖 Simulador Predictivo (ML)"])

with tab1:
    # CORRECCIÓN DE PANDAS: Se usa 'ME' (Month End)
    df_temporal = df_filtered.set_index('Fecha').resample('ME').mean(numeric_only=True).reset_index()
    fig_line = px.line(
        df_temporal, x='Fecha', y='PM2.5 (µg/m³)', 
        title='Evolución Mensual de la Contaminación (PM2.5)',
        markers=True, template='plotly_white', color_discrete_sequence=['#E63946']
    )
    # Añadir línea de límite recomendado por la OMS
    fig_line.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Límite OMS")
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_scatter1 = px.scatter(
            df_filtered, x='Tráfico (Vehículos)', y='PM2.5 (µg/m³)', 
            trendline="ols", title='Impacto del Tráfico en la Contaminación',
            template='plotly_white', opacity=0.5
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    with col_b:
        fig_scatter2 = px.scatter(
            df_filtered, x='Viento (km/h)', y='PM2.5 (µg/m³)', 
            trendline="ols", title='Efecto Dispersor del Viento',
            template='plotly_white', opacity=0.5, color_discrete_sequence=['#2A9D8F']
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)

with tab3:
    st.markdown("### 🧠 Modelo de Regresión Lineal Múltiple")
    st.write("El modelo ha aprendido la relación entre el tráfico, el clima y la contaminación. Modifica los valores a continuación para simular el nivel de PM2.5 esperado.")
    
    # Entrenar el modelo con los datos históricos
    X = df_filtered[['Tráfico (Vehículos)', 'Viento (km/h)', 'Temperatura (°C)']]
    y = df_filtered['PM2.5 (µg/m³)']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Interfaz interactiva para el usuario (What-If Analysis)
    st.markdown("#### 🎛️ Simulador de Escenarios Urbanos")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        input_trafico = st.slider("Tráfico Vehicular (veh/día)", 10000, 120000, int(trafico_promedio), step=5000)
    with sim_col2:
        input_viento = st.slider("Velocidad del Viento (km/h)", 0, 40, int(viento_promedio), step=1)
    with sim_col3:
        input_temp = st.slider("Temperatura (°C)", -5, 40, int(df_filtered['Temperatura (°C)'].mean()), step=1)
        
    # Hacer la predicción
    prediccion = model.predict([[input_trafico, input_viento, input_temp]])[0]
    prediccion = max(0, prediccion)
    
    # Mostrar resultado de la simulación
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Predicción de PM2.5", f"{prediccion:.1f} µg/m³", 
                  delta="Peligroso" if prediccion > 25 else "Moderado" if prediccion > 15 else "Bueno",
                  delta_color="inverse")
    with res_col2:
        st.info(f"El modelo estima que con **{input_trafico:,} vehículos** y vientos de **{input_viento} km/h**, la calidad del aire será {'**MALA** (se recomienda restringir circulación)' if prediccion > 25 else '**ACEPTABLE**'}.")

# 6. Tabla de datos y descarga
st.subheader("📋 Datos Históricos")
st.dataframe(
    df_filtered.sort_values(by='Fecha', ascending=False).head(100),
    use_container_width=True
)

csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Datos Ambientales (CSV)",
    data=csv, file_name='calidad_aire_smartcity.csv', mime='text/csv', type="primary"
)

st.sidebar.markdown("---")
st.sidebar.success("✅ Modelo de IA entrenado y listo.")