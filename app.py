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

# ---------------- CARGA AUTOMÁTICA DEL CSV ----------------
CSV_PATH = "data/AmesHousing.csv"

try:
    df = pd.read_csv(CSV_PATH)
    st.sidebar.info(f"Cargando CSV automáticamente desde `{CSV_PATH}`")
except Exception as e:
    st.error(f"❌ No se pudo cargar el CSV: {e}")
    st.stop()

# ---------------- VERIFICACIÓN DE COLUMNAS ----------------
columnas_necesarias = ["SalePrice", "Neighborhood", "Year Built",
                       "Gr Liv Area", "Overall Qual", "Garage Cars"]
if not all(col in df.columns for col in columnas_necesarias):
    st.error(f"⚠️ El archivo no contiene las columnas necesarias: {columnas_necesarias}")
    st.stop()

# Convertir columnas a tipo numérico por seguridad
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
st.pyplot(fig)

# Área habitable vs Precio
st.subheader("Área habitable vs Precio de Venta")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x="Gr Liv Area", y="SalePrice", data=df, alpha=0.6, ax=ax)
st.pyplot(fig)

# Precio por calidad general
st.subheader("Precio de Venta por Calidad General")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="Overall Qual", y="SalePrice", data=df, ax=ax)
st.pyplot(fig)

# Precio promedio por vecindario
st.subheader("Precio Promedio por Vecindario")
neigh_price = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=neigh_price.index, y=neigh_price.values, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Correlación con el precio
st.subheader("Correlación con el Precio de Venta")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(
    x=corr["SalePrice"].sort_values(ascending=False).values,
    y=corr["SalePrice"].sort_values(ascending=False).index,
    ax=ax
)
ax.set_title("Correlación con SalePrice")
st.pyplot(fig)

# Mapa de calor
st.write("Mapa de calor de correlaciones:")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
st.pyplot(fig)

st.markdown("---")

# ---------------- MODELO DE PREDICCIÓN ----------------
st.subheader("Modelo de Predicción (Regresión Lineal)")

X = df[["Gr Liv Area", "Overall Qual", "Garage Cars", "Year Built"]]
y = df["SalePrice"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

c1, c2 = st.columns(2)
with c1:
    st.metric("Error Absoluto Medio", f"${mae:,.0f}")
with c2:
    st.metric("R²", f"{r2:.2f}")

fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
ax.set_xlabel("Precio Real")
ax.set_ylabel("Precio Predicho")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
st.pyplot(fig)

st.markdown("---")

# ---------------- SIMULADOR ----------------
st.subheader("🔮 Simulador de Precio de Vivienda")
area = st.slider("Área habitable (Gr Liv Area)", 300, 6000, 1500)
calidad = st.slider("Calidad General (Overall Qual)", 1, 10, 5)
garage = st.slider("Número de Coches en Garage", 0, 4, 2)
anio = st.slider("Año de Construcción", 1870, 2025, 2000)

entrada = pd.DataFrame([[int(area), int(calidad), int(garage), int(anio)]],
                       columns=["Gr Liv Area", "Overall Qual", "Garage Cars", "Year Built"])
entrada_imputada = imputer.transform(entrada)
prediccion = model.predict(entrada_imputada)[0]
st.success(f"💰 Precio estimado de la vivienda: **${prediccion:,.0f}**")

st.subheader("Proyección de Precios a Futuro")
inflacion = st.slider("Tasa de Inflación Anual (%)", 0.0, 10.0, 3.0) / 100
anio_futuro = st.slider("Año de proyección", 2023, 2050, 2025)
precio_futuro = prediccion * ((1 + inflacion) ** (anio_futuro - 2023))
st.info(f"📈 Precio estimado en {anio_futuro}: **${precio_futuro:,.0f}**")
