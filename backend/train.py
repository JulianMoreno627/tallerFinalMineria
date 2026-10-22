"""
Entrenamiento del modelo K-Means para segmentación de clientes.

Uso (desde la raíz del proyecto):
    python backend/train.py
    python backend/train.py --data datos/raw/otro_archivo.csv
    python backend/train.py --k 5
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = ROOT / "datos" / "raw" / "Mall_Customers.csv"
PROCESSED_DIR = ROOT / "datos" / "processed"
MODEL_DIR = ROOT / "modelo"
IMG_DIR = ROOT / "imagenes"

FEATURES = ["Ingreso_Anual_k", "Puntaje_Gasto"]
NUM_COLS = ["Edad", "Ingreso_Anual_k", "Puntaje_Gasto"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def cargar(ruta: Path) -> pd.DataFrame:
    print(f"[1] Cargando dataset: {ruta}")
    df = pd.read_csv(ruta)
    print(f"    {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("[2] Limpiando datos")
    reporte = {"filas_iniciales": int(len(df))}
    df = df.copy()

    # Renombrar columnas
    df.columns = [c.strip() for c in df.columns]
    renombres = {
        "CustomerID": "ID_Cliente",
        "Gender": "Genero",
        "Genre": "Genero",  # algunas versiones del dataset usan "Genre"
        "Age": "Edad",
        "Annual Income (k$)": "Ingreso_Anual_k",
        "Spending Score (1-100)": "Puntaje_Gasto",
    }
    df = df.rename(columns=renombres)
    faltantes = set(["Genero"] + NUM_COLS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Al dataset le faltan columnas: {faltantes}")

    # Normalizar género
    df["Genero"] = (df["Genero"].astype(str).str.strip().str.capitalize()
                    .replace({"M": "Male", "F": "Female", "Masculino": "Male",
                              "Hombre": "Male", "Femenino": "Female", "Mujer": "Female"}))

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Duplicados
    dup = int(df.duplicated(subset=[c for c in df.columns if c != "ID_Cliente"]).sum()
              if "ID_Cliente" in df.columns else df.duplicated().sum())
    if "ID_Cliente" in df.columns:
        df = df.drop_duplicates(subset="ID_Cliente")
    df = df.drop_duplicates()
    reporte["duplicados_eliminados"] = reporte["filas_iniciales"] - int(len(df))
    reporte["duplicados_detectados"] = dup

    # Nulos
    reporte["nulos_por_columna"] = {k: int(v) for k, v in df.isnull().sum().items()}
    antes = len(df)
    df = df.dropna(subset=NUM_COLS + ["Genero"])
    reporte["filas_con_nulos_eliminadas"] = int(antes - len(df))

    # Valores fuera de rango
    antes = len(df)
    df = df[(df["Edad"].between(1, 120)) &
            (df["Ingreso_Anual_k"] > 0) &
            (df["Puntaje_Gasto"].between(1, 100))]
    reporte["filas_fuera_de_rango_eliminadas"] = int(antes - len(df))

    # Atípicos por IQR (solo se marcan)
    atipicos = {}
    df["Es_Atipico"] = False
    for col in NUM_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lim_inf) | (df[col] > lim_sup)
        df.loc[mask, "Es_Atipico"] = True
        atipicos[col] = {"cantidad": int(mask.sum()),
                         "limite_inferior": round(float(lim_inf), 2),
                         "limite_superior": round(float(lim_sup), 2)}
    reporte["atipicos_iqr"] = atipicos

    for col in NUM_COLS:
        df[col] = df[col].astype(int)
    if "ID_Cliente" in df.columns:
        df["ID_Cliente"] = df["ID_Cliente"].astype(int)
    df = df.reset_index(drop=True)
    reporte["filas_finales"] = int(len(df))

    for k, v in reporte.items():
        print(f"    {k}: {v}")
    return df, reporte


def evaluar_k(X_scaled, k_max=10):
    print("[4] Evaluando K (codo y silueta)")
    filas = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_) if k > 1 else None
        filas.append({"k": k, "inercia": round(float(km.inertia_), 4),
                      "silueta": None if sil is None else round(float(sil), 4)})
        print(f"    K={k:2d}  inercia={km.inertia_:8.2f}  silueta={'-' if sil is None else f'{sil:.4f}'}")
    return filas


def nombrar_perfil(ingreso, gasto):
    if ingreso >= 70 and gasto >= 60:
        return "Alto valor"
    if ingreso >= 70 and gasto < 40:
        return "Con potencial"
    if ingreso < 40 and gasto < 40:
        return "Bajo valor"
    if ingreso < 40 and gasto >= 60:
        return "Gastadores impulsivos"
    return "Promedio"


ESTRATEGIAS = {
    "Alto valor": "Fidelización: programa VIP, acceso anticipado a lanzamientos, atención personalizada y eventos exclusivos.",
    "Con potencial": "Activación: productos premium y experiencias, campañas personalizadas e incentivos para aumentar su gasto.",
    "Promedio": "Aumentar el ticket: programa de puntos, venta cruzada, combos y promociones de temporada.",
    "Gastadores impulsivos": "Redes sociales, tendencias y moda, pago en cuotas y recompensas por frecuencia (cuidando el riesgo crediticio).",
    "Bajo valor": "Descuentos, productos básicos y comunicación masiva de bajo costo.",
}
COLORES = {"Alto valor": "#2ca02c", "Con potencial": "#ff7f0e", "Promedio": "#1f77b4",
           "Gastadores impulsivos": "#9467bd", "Bajo valor": "#d62728"}


def guardar(fig, nombre):
    ruta = IMG_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    imagen -> {ruta.relative_to(ROOT)}")


def graficas_exploracion(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, NUM_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribución de {col}")
    guardar(fig, "01_distribuciones.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, NUM_COLS):
        sns.boxplot(y=df[col], ax=ax, color="lightcoral")
        ax.set_title(col)
    guardar(fig, "02_boxplots_atipicos.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(df[NUM_COLS].corr(), annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de correlación")
    guardar(fig, "03_correlacion.png")

    g = sns.pairplot(df, vars=NUM_COLS, hue="Genero", diag_kind="kde", palette="Set2")
    g.savefig(IMG_DIR / "04_pairplot.png", dpi=120)
    plt.close(g.figure)
    print("    imagen -> imagenes/04_pairplot.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], s=60, ax=ax)
    ax.set_title("Ingreso anual vs. Puntaje de gasto")
    guardar(fig, "05_dispersion_ingreso_gasto.png")


def graficas_k(evaluacion, k):
    ks = [e["k"] for e in evaluacion]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [e["inercia"] for e in evaluacion], "o-", color="steelblue", lw=2)
    ax.axvline(k, color="red", ls="--", label=f"K = {k}")
    ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo"); ax.set_xticks(ks); ax.legend()
    guardar(fig, "06_metodo_codo.png")

    ev = [e for e in evaluacion if e["silueta"] is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([e["k"] for e in ev], [e["silueta"] for e in ev], "o-", color="seagreen", lw=2)
    ax.axvline(k, color="red", ls="--", label=f"K = {k}")
    ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Coeficiente de silueta")
    ax.set_title("Coeficiente de silueta por K"); ax.set_xticks([e["k"] for e in ev]); ax.legend()
    guardar(fig, "07_coeficiente_silueta.png")


def graficas_modelo(df, X_scaled, model, centroides):
    k = model.n_clusters
    sil_vals = silhouette_samples(X_scaled, model.labels_)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_low = 10
    for c in range(k):
        vals = np.sort(sil_vals[model.labels_ == c])
        ax.fill_betweenx(np.arange(y_low, y_low + len(vals)), 0, vals, alpha=0.7)
        ax.text(-0.05, y_low + len(vals) / 2, str(c))
        y_low += len(vals) + 10
    ax.axvline(sil_vals.mean(), color="red", ls="--", label=f"Promedio = {sil_vals.mean():.3f}")
    ax.set_xlabel("Coeficiente de silueta"); ax.set_yticks([])
    ax.set_title(f"Silueta por observación (K = {k})"); ax.legend()
    guardar(fig, "08_silueta_por_observacion.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Cluster", palette="tab10", s=70, ax=ax)
    ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", marker="X", s=250, label="Centroides")
    ax.set_title(f"Clusters de clientes (K = {k})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    guardar(fig, "09_clusters.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    pal = {p: COLORES.get(p, "gray") for p in df["Perfil"].unique()}
    sns.scatterplot(data=df, x=FEATURES[0], y=FEATURES[1], hue="Perfil", palette=pal, s=70, ax=ax)
    ax.scatter(centroides[FEATURES[0]], centroides[FEATURES[1]], c="black", marker="X", s=250)
    ax.set_title("Perfiles de negocio")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    guardar(fig, "10_perfiles_negocio.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, NUM_COLS):
        sns.boxplot(data=df, x="Cluster", y=col, hue="Cluster", palette="tab10", legend=False, ax=ax)
        ax.set_title(col)
    guardar(fig, "11_variables_por_cluster.png")


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo K-Means de segmentación de clientes")
    parser.add_argument("--data", type=str, default=str(RAW_DEFAULT), help="Ruta al CSV crudo")
    parser.add_argument("--k", type=int, default=None, help="Número de clusters (por defecto: el de mayor silueta)")
    parser.add_argument("--kmax", type=int, default=10, help="K máximo a evaluar")
    args = parser.parse_args()

    for d in (PROCESSED_DIR, MODEL_DIR, IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df_raw = cargar(Path(args.data))
    df, reporte = limpiar(df_raw)
    ruta_limpio = PROCESSED_DIR / "Mall_Customers_limpio.csv"
    df.to_csv(ruta_limpio, index=False)
    print(f"    dataset limpio -> {ruta_limpio.relative_to(ROOT)}")
    with open(PROCESSED_DIR / "reporte_limpieza.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("[*] Generando gráficas de exploración")
    graficas_exploracion(df)
    print("[3] Escalando variables con StandardScaler:", FEATURES)
    X = df[FEATURES]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    evaluacion = evaluar_k(X_scaled, args.kmax)
    k_silueta = max((e for e in evaluacion if e["silueta"] is not None), key=lambda e: e["silueta"])["k"]
    k = args.k or k_silueta
    print(f"    K con mayor silueta: {k_silueta}  ->  K usado: {k}")
    graficas_k(evaluacion, k)
    print(f"[5] Entrenando K-Means con K = {k}")
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    sil = float(silhouette_score(X_scaled, model.labels_))
    print(f"    inercia = {model.inertia_:.2f}   silueta = {sil:.4f}")

    df["Cluster"] = model.labels_
    centroides = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    perfiles = {int(c): nombrar_perfil(*centroides.loc[c]) for c in centroides.index}
    df["Perfil"] = df["Cluster"].map(perfiles)

    resumen = df.groupby("Cluster").agg(
        Perfil=("Perfil", "first"),
        Clientes=("Edad", "count"),
        Edad_media=("Edad", "mean"),
        Ingreso_medio=("Ingreso_Anual_k", "mean"),
        Gasto_medio=("Puntaje_Gasto", "mean"),
        Pct_mujeres=("Genero", lambda g: (g == "Female").mean() * 100),
    ).round(1)
    print(resumen.to_string())

    df.to_csv(PROCESSED_DIR / "Mall_Customers_clusters.csv", index=False)
    resumen.to_csv(PROCESSED_DIR / "resumen_clusters.csv")
    print("[6] Exportando modelo")
    joblib.dump(model, MODEL_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "dataset": Path(args.data).name,
        "features": FEATURES,
        "k": int(k),
        "k_mayor_silueta": int(k_silueta),
        "inercia": round(float(model.inertia_), 4),
        "silueta": round(sil, 4),
        "evaluacion_k": evaluacion,
        "centroides": centroides.round(2).to_dict(orient="index"),
        "perfiles": perfiles,
        "estrategias": ESTRATEGIAS,
        "colores": COLORES,
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    for n in ("kmeans_model.joblib", "scaler.joblib", "metadata.json"):
        print(f"    -> modelo/{n}")
    print("[7] Generando gráficas del modelo")
    graficas_modelo(df, X_scaled, model, centroides)
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
