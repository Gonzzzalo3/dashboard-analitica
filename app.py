import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. Configuración de la página
st.set_page_config(
    page_title="Dashboard de Analítica Comercial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para un diseño limpio (Tema abierto/claro)
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Dashboard de Analítica Comercial Avanzada")
st.markdown("---")

# 2. Generación de datos de negocio simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    fechas = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes']
    regiones = ['Norte', 'Sur', 'Este', 'Oeste']
    
    data = []
    for _ in range(1000):
        fecha = np.random.choice(fechas)
        categoria = np.random.choice(categorias)
        region = np.random.choice(regiones)
        ventas = np.random.uniform(50, 1000)
        margen = ventas * np.random.uniform(0.15, 0.45)
        data.append([fecha, categoria, region, ventas, margen])
        
    df = pd.DataFrame(data, columns=['Fecha', 'Categoría', 'Región', 'Ventas (USD)', 'Margen (USD)'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

df_ventas = load_data()

# 3. Barra lateral con filtros interactivos
st.sidebar.header("🔍 Filtros de Análisis")
regiones_sel = st.sidebar.multiselect(
    "Selecciona Región:", 
    options=df_ventas['Región'].unique(), 
    default=df_ventas['Región'].unique()
)
categorias_sel = st.sidebar.multiselect(
    "Selecciona Categoría:", 
    options=df_ventas['Categoría'].unique(), 
    default=df_ventas['Categoría'].unique()
)

# Filtrar DataFrame
df_filtered = df_ventas[
    (df_ventas['Región'].isin(regiones_sel)) & 
    (df_ventas['Categoría'].isin(categorias_sel))
]

# 4. Métricas clave (KPIs)
st.subheader("💡 Resumen Ejecutivo")
col1, col2, col3, col4 = st.columns(4)

ventas_totales = df_filtered['Ventas (USD)'].sum()
margen_total = df_filtered['Margen (USD)'].sum()
ticket_promedio = df_filtered['Ventas (USD)'].mean()
num_transacciones = len(df_filtered)

with col1:
    st.metric("Ventas Totales", f"${ventas_totales:,.2f}")
with col2:
    st.metric("Margen Total", f"${margen_total:,.2f}")
with col3:
    st.metric("Ticket Promedio", f"${ticket_promedio:,.2f}")
with col4:
    st.metric("Total Transacciones", f"{num_transacciones:,}")

st.markdown("---")

# 5. Visualizaciones interactivas con Plotly
st.subheader("📈 Tendencias y Distribución")

tab1, tab2, tab3 = st.tabs(["Evolución Temporal", "Distribución por Categoría", "Análisis por Región"])

with tab1:
    df_temporal = df_filtered.set_index('Fecha').resample('M').sum().reset_index()
    fig_line = px.line(
        df_temporal, 
        x='Fecha', 
        y='Ventas (USD)', 
        title='Evolución Mensual de Ventas',
        markers=True,
        template='plotly_white'
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    fig_bar = px.bar(
        df_filtered.groupby('Categoría')['Ventas (USD)'].sum().reset_index(),
        x='Categoría',
        y='Ventas (USD)',
        color='Categoría',
        title='Ventas por Categoría de Producto',
        template='plotly_white'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    fig_pie = px.pie(
        df_filtered,
        names='Región',
        values='Ventas (USD)',
        title='Participación de Ventas por Región',
        template='plotly_white',
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# 6. Análisis detallado y descarga
st.subheader("📋 Detalle de Transacciones")
st.dataframe(
    df_filtered[['Fecha', 'Categoría', 'Región', 'Ventas (USD)', 'Margen (USD)']]
    .sort_values(by='Ventas (USD)', ascending=False)
    .head(50),
    use_container_width=True
)

# Botón de descarga
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Datos Filtrados (CSV)",
    data=csv,
    file_name='datos_comerciales.csv',
    mime='text/csv',
    type="primary"
)

st.sidebar.markdown("---")
st.sidebar.success("✅ Dashboard desplegado con éxito.")