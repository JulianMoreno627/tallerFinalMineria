# Segmentación de clientes con K-Means – Taller Final Minería de Datos

Proyecto de segmentación de clientes del *Mall Customers Dataset* usando K-Means, método del codo y coeficiente de silueta, con despliegue en Streamlit.

## Estructura

```
proyecto_kmeans/
├── datos/
│   ├── raw/                         # Dataset original sin procesar ("sucio")
│   │   └── Mall_Customers.csv
│   └── processed/                   # Lo genera el backend
│       ├── Mall_Customers_limpio.csv      # dataset limpio
│       ├── Mall_Customers_clusters.csv    # dataset limpio + cluster y perfil
│       ├── resumen_clusters.csv           # perfil promedio de cada cluster
│       └── reporte_limpieza.json          # qué se limpió (nulos, duplicados, atípicos…)
├── backend/
│   └── train.py                     # carga, limpia, entrena y exporta el modelo
├── modelo/                          # Lo genera el backend
│   ├── kmeans_model.joblib          # modelo K-Means entrenado
│   ├── scaler.joblib                # StandardScaler ajustado
│   └── metadata.json                # K, métricas, centroides, perfiles
├── imagenes/                        # Lo genera el backend (codo, silueta, clusters…)
├── frontend/
│   └── app.py                       # aplicación Streamlit
├── notebook/
│   └── Taller_KMeans_Segmentacion.ipynb   # desarrollo y respuestas del taller
├── requirements.txt
└── README.md
```

## Cómo ejecutar

Abrir una terminal **en la carpeta raíz del proyecto** (`proyecto_kmeans`):

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar el modelo (limpia datos, genera imágenes y exporta el .joblib)
python backend/train.py

# 3. Ejecutar la aplicación
streamlit run frontend/app.py
```

Si `streamlit` no se reconoce como comando (común en Windows):
```bash
python -m streamlit run frontend/app.py
```

### Opciones del backend

```bash
python backend/train.py --data ruta/a/otro_dataset.csv   # usar otro CSV
python backend/train.py --k 6                            # forzar un K
python backend/train.py --kmax 15                        # evaluar hasta K = 15
```
Por defecto el K se elige automáticamente como el de mayor coeficiente de silueta.

## Limpieza que realiza el backend

1. Renombra columnas (`Annual Income (k$)` → `Ingreso_Anual_k`, etc.).
2. Normaliza el género (espacios, mayúsculas, variantes en español).
3. Convierte las columnas numéricas (texto inválido → nulo).
4. Elimina duplicados.
5. Elimina filas con nulos.
6. Elimina valores imposibles (edad fuera de 1–120, puntaje fuera de 1–100, ingreso ≤ 0).
7. Detecta atípicos con IQR y los **marca** en la columna `Es_Atipico` (no los elimina).

## Resultados

| Cluster | Perfil | Clientes | Ingreso medio | Gasto medio |
|---|---|---|---|---|
| 1 | Alto valor | 39 | 86.5 k$ | 82.1 |
| 3 | Con potencial | 35 | 88.2 k$ | 17.1 |
| 0 | Promedio | 81 | 55.3 k$ | 49.5 |
| 2 | Gastadores impulsivos | 22 | 25.7 k$ | 79.4 |
| 4 | Bajo valor | 23 | 26.3 k$ | 20.9 |

K óptimo = **5** (codo y silueta coinciden; silueta = 0.555).

## Despliegue en Streamlit Community Cloud

1. Subir la carpeta completa del proyecto a un repositorio público de GitHub (incluyendo `modelo/`, `datos/` e `imagenes/`).
2. Entrar a https://share.streamlit.io, elegir **Create app** y seleccionar el repositorio.
3. En *Main file path* escribir `frontend/app.py` y hacer clic en **Deploy**.
