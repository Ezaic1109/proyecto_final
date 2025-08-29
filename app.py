import os
from dotenv import load_dotenv
import mysql.connector
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Cargar variables desde .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )


# ========================================
# CARGA DE DATOS
# ========================================
@st.cache_data
def load_data():
    conn = get_connection()
    query = "SELECT * FROM fact_taxi_trips;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ========================================
# CARGA DE DATOS
# ========================================
#@st.cache_data
#def load_data():
#   conn = get_connection()
#   query = "SELECT * FROM fact_taxi_trips;"
#   df = pd.read_sql(query, conn)
#   conn.close()
#   return df

# ========================================
# APP STREAMLIT
# ========================================
#st.set_page_config(page_title="NYC Taxi Dashboard", layout="wide")

#st.title("🚕 NYC Taxi Dashboard")
#st.markdown("Visualización dinámica de la tabla `fact_taxi_trips` con **matplotlib/seaborn**")

# Cargar datos
#df = load_data()

# ==============================
# FILTROS DINÁMICOS
# ==============================
#col1, col2 = st.columns(2)

#with col1:
#   years = st.multiselect("Selecciona año(s):", options=df["year"].unique(), default=df["year"].unique())

#with col2:
 #  payment_types = st.multiselect("Método de pago:", options=df["payment_type"].unique(), default=df["payment_type"].unique())

# Filtrar dataset
#filtered_df = df[(df["year"].isin(years)) & (df["payment_type"].isin(payment_types))]

#st.dataframe(filtered_df)

# ==============================
# GRÁFICAS
# ==============================


#st.subheader("💵 Ingresos totales por día")
#revenue_by_day = df.groupby("pickup_date")["total_amount"].sum().reset_index()

#fig, ax = plt.subplots()
#sns.lineplot(data=revenue_by_day, x="pickup_date", y="total_amount", marker="o", ax=ax)

#ax.set_title("Ingresos Totales por Día")
#ax.set_xlabel("Fecha")
#ax.set_ylabel("Total ($)")

#ax.xaxis.set_major_locator(mdates.AutoDateLocator())     # detecta mejor intervalos
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # formato YYYY-MM-DD
#plt.xticks(rotation=45, ha="right")  # rota etiquetas

#st.pyplot(fig)

#st.subheader("💳 Propinas promedio por método de pago")
#tips_by_payment = (
#   filtered_df.groupby("payment_type")["tip_amount"].mean().reset_index()
#)

#fig, ax = plt.subplots()
#sns.barplot(data=tips_by_payment, x="payment_type", y="tip_amount", ax=ax)
#ax.set_title("Propina promedio por método de pago")
#ax.set_xlabel("Método de pago")
#ax.set_ylabel("Propina promedio ($)")
#st.pyplot(fig)

#st.subheader("🚏 Distancia promedio por día de la semana")
#dist_by_day = (
#    filtered_df.groupby("day_of_week")["trip_distance"].mean().reset_index()
#)
#fig, ax = plt.subplots()
#sns.barplot(data=dist_by_day, x="day_of_week", y="trip_distance", ax=ax)
#ax.set_title("Distancia promedio por día de la semana")
#ax.set_xlabel("Día de la semana")
#ax.set_ylabel("Distancia promedio (millas)")
#st.pyplot(fig)
# ========================================
# STREAMLIT APP
# ========================================

st.set_page_config(page_title="🏠 AmesHousing Dashboard", layout="wide")
st.title("🏠 Análisis del Mercado Inmobiliario en Ames, Iowa")
st.markdown("Explora el dataset de AmesHousing, descubre patrones y estima precios de viviendas 📊")
st.markdown("---")

# ========================================
# CARGAR CSV AUTOMÁTICAMENTE
# ========================================
st.sidebar.title("Cargar Datos")
st.sidebar.markdown("CSV cargado automáticamente desde el contenedor.")
CSV_PATH = "data/AmesHousing.csv"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

try:
    df = load_data(CSV_PATH)
    st.sidebar.info(f"Cargando CSV automáticamente desde `{CSV_PATH}`")
except Exception as e:
    st.error(f"❌ No se pudo cargar el CSV: {e}")
    st.stop()

# ========================================
# VERIFICACIÓN DE COLUMNAS
# ========================================
columnas_necesarias = ["SalePrice", "Neighborhood", "Year Built",
                       "Gr Liv Area", "Overall Qual", "Garage Cars"]
if not all(col in df.columns for col in columnas_necesarias):
    st.error(f"⚠️ El archivo no contiene las columnas necesarias: {columnas_necesarias}")
    st.stop()

# Convertir columnas a tipo numérico
df[["SalePrice","Gr Liv Area","Overall Qual","Garage Cars","Year Built"]] = \
    df[["SalePrice","Gr Liv Area","Overall Qual","Garage Cars","Year Built"]].apply(pd.to_numeric, errors='coerce')

# ---------------- FILTROS ----------------
st.sidebar.subheader("Filtros")
barrios = ["Todos"] + sorted(df["Neighborhood"].dropna().unique().tolist())
barrio_sel = st.sidebar.selectbox("Selecciona un vecindario:", barrios)
if barrio_sel != "Todos":
    df = df[df["Neighborhood"] == barrio_sel].copy()
    st.success(f"Analizando solo el vecindario: **{barrio_sel}**")

# ---------------- VISTA PREVIA ----------------
st.subheader("Vista Previa de los Datos")
st.dataframe(df.head())

# ---------------- MÉTRICAS ----------------
st.subheader("Métricas Clave")
total_casas = len(df)
precio_prom = df["SalePrice"].mean()
anio_min, anio_max = df["Year Built"].min(), df["Year Built"].max()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total de Casas", total_casas)
with c2:
    st.metric("Precio Promedio", f"${precio_prom:,.0f}")
with c3:
    st.metric("Rango de Construcción", f"{anio_min} - {anio_max}")

st.markdown("---")

# ---------------- VISUALIZACIONES ----------------
# Distribución de precios
st.subheader("Distribución de Precios de Venta")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df["SalePrice"], bins=30, kde=True, ax=ax)
fig.tight_layout()
st.pyplot(fig)

# Área habitable vs Precio
st.subheader("Área habitable vs Precio de Venta")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x="Gr Liv Area", y="SalePrice", data=df, alpha=0.6, ax=ax)
fig.tight_layout()
st.pyplot(fig)

# Precio por calidad general
st.subheader("Precio de Venta por Calidad General")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="Overall Qual", y="SalePrice", data=df, ax=ax)
fig.tight_layout()
st.pyplot(fig)

# Precio promedio por vecindario
st.subheader("Precio Promedio por Vecindario")
neigh_price = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=neigh_price.index, y=neigh_price.values, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
st.pyplot(fig)
