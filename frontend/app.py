import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "modelo"
DATA_DIR = ROOT / "datos" / "processed"
IMG_DIR = ROOT / "imagenes"

st.set_page_config(page_title="Segmentación de clientes", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_resource
def cargar_modelo():
    model = joblib.load(MODEL_DIR / "kmeans_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    meta["perfiles"] = {int(k): v for k, v in meta["perfiles"].items()}
    return model, scaler, meta


@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_DIR / "Mall_Customers_clusters.csv")


if not (MODEL_DIR / "kmeans_model.joblib").exists():
    st.error("No se encontró el modelo entrenado. Ejecuta primero:  `python backend/train.py`")
    st.stop()

model, scaler, meta = cargar_modelo()
df = cargar_datos()
FEATURES = meta["features"]
K = meta["k"]
perfiles = meta["perfiles"]
evaluacion = pd.DataFrame(meta["evaluacion_k"])
centroides = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
colores = meta["colores"]

st.title("Segmentación de clientes con K-Means")
st.caption(f"Modelo entrenado el {meta['fecha_entrenamiento'].replace('T', ' ')} con {meta['dataset']} · "
           f"variables: ingreso anual y puntaje de gasto")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clientes", len(df))
m2.metric("Clusters (K)", K)
m3.metric("Inercia (WCSS)", f"{meta['inercia']:.2f}")
m4.metric("Coef. de silueta", f"{meta['silueta']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Método del codo", "Clusters", "Nuevo cliente", "Galería", "Datos"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(evaluacion["k"], evaluacion["inercia"], "o-", color="steelblue", lw=2)
        ax.axvline(K, color="red", ls="--", label=f"K = {K}")
        ax.set_xlabel("K"); ax.set_ylabel("Inercia (WCSS)"); ax.set_title("Método del codo")
        ax.set_xticks(evaluacion["k"]); ax.legend()
        st.pyplot(fig)
    with c2:
        ev = evaluacion.dropna(subset=["silueta"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ev["k"], ev["silueta"], "o-", color="seagreen", lw=2)
        ax.axvline(K, color="red", ls="--", label=f"K = {K}")
        ax.set_xlabel("K"); ax.set_ylabel("Silueta promedio"); ax.set_title("Coeficiente de silueta")
        ax.set_xticks(ev["k"]); ax.legend()
        st.pyplot(fig)
    st.dataframe(evaluacion.rename(columns={"k": "K", "inercia": "Inercia", "silueta": "Silueta"}),
                 hide_index=True)
    st.info(f"El codo se forma en K = {K} y la silueta es máxima en K = {meta['k_mayor_silueta']} "
            f"({meta['silueta']:.3f}).")

with tab2:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Perfil",
                        palette={p: colores.get(p, "gray") for p in df["Perfil"].unique()}, s=60, ax=ax)
        ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", marker="X", s=220,
                   label="Centroides")
        ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Puntaje de gasto (1-100)")
        ax.set_title(f"Clusters generados (K = {K})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        st.pyplot(fig)
    with c2:
        st.subheader("Perfil de cada cluster")
        resumen = df.groupby("Cluster").agg(
            Perfil=("Perfil", "first"),
            Clientes=("Edad", "count"),
            Edad=("Edad", "mean"),
            Ingreso=("Ingreso_Anual_k", "mean"),
            Gasto=("Puntaje_Gasto", "mean"),
        ).round(1)
        st.dataframe(resumen)
        for c, p in perfiles.items():
            st.markdown(f"**Cluster {c} – {p}:** {meta['estrategias'][p]}")

with tab3:
    st.subheader("Clasificar un nuevo cliente")
    with st.form("nuevo_cliente"):
        c1, c2, c3, c4 = st.columns(4)
        genero = c1.selectbox("Género", ["Female", "Male"])
        edad = c2.number_input("Edad", 18, 100, 30)
        ingreso = c3.number_input("Ingreso anual (k$)", 1, 300, 60)
        gasto = c4.slider("Puntaje de gasto (1-100)", 1, 100, 50)
        enviar = st.form_submit_button("Clasificar cliente")

    if enviar:
        nuevo = pd.DataFrame([[ingreso, gasto]], columns=FEATURES)
        nuevo_esc = scaler.transform(nuevo)
        cluster = int(model.predict(nuevo_esc)[0])
        perfil = perfiles[cluster]
        dist = np.linalg.norm(model.cluster_centers_ - nuevo_esc, axis=1)

        r1, r2 = st.columns([2, 3])
        with r1:
            st.success(f"El cliente pertenece al **cluster {cluster}**")
            st.markdown(f"### {perfil}")
            st.markdown(f"**Estrategia sugerida:** {meta['estrategias'][perfil]}")
            st.caption("Distancia (escalada) a cada centroide:")
            st.dataframe(pd.DataFrame({"Cluster": range(K),
                                       "Perfil": [perfiles[i] for i in range(K)],
                                       "Distancia": dist.round(3)}), hide_index=True)
        with r2:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Perfil",
                            palette={p: colores.get(p, "gray") for p in df["Perfil"].unique()},
                            s=40, alpha=0.45, ax=ax)
            ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", marker="X", s=180)
            ax.scatter(ingreso, gasto, c="red", marker="*", s=600, edgecolor="black", label="Nuevo cliente")
            ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Puntaje de gasto (1-100)")
            ax.set_title("Ubicación del nuevo cliente")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            st.pyplot(fig)

with tab4:
    imagenes = sorted(IMG_DIR.glob("*.png"))
    if not imagenes:
        st.warning("No hay imágenes. Ejecuta `python backend/train.py`.")
    for i in range(0, len(imagenes), 2):
        cols = st.columns(2)
        for col, img in zip(cols, imagenes[i:i + 2]):
            col.image(str(img), caption=img.stem[3:].replace("_", " ").capitalize())

with tab5:
    st.dataframe(df)
    st.download_button("Descargar datos con clusters", df.to_csv(index=False).encode("utf-8"),
                       "Mall_Customers_clusters.csv", "text/csv")
