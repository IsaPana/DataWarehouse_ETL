import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 1. Cargar dataset

df = pd.read_csv("Morning_Routine_Productivity_CLEAN.csv")

# Seleccionar columnas relevantes
numeric_cols = ["sleep_duration_hrs", "meditation_mins", "exercise_mins", "Journaling"]
cat_cols = ["breakfast_type", "mood"]


# 2. Preprocesamiento (OneHot + Escalado)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

X = preprocessor.fit_transform(df)


# 3. Reducción PCA a 2 componentes para visualización

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# 4. Evaluación de varios valores de k

resultados = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    etiquetas = kmeans.fit_predict(X)

    sil = silhouette_score(X, etiquetas)
    ch = calinski_harabasz_score(X, etiquetas)
    db = davies_bouldin_score(X, etiquetas)

    resultados.append([k, sil, ch, db])

resultados_df = pd.DataFrame(resultados, columns=["k", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])

print("\n=== RESULTADOS DE CLUSTERING ===")
print(resultados_df)

# Seleccionar mejor k por Silhouette
mejor_k = resultados_df.iloc[resultados_df["Silhouette"].idxmax()]["k"]
print(f"\nMejor k según Silhouette Score: {int(mejor_k)}")

# Entrenar modelo final
kmeans_final = KMeans(n_clusters=int(mejor_k), random_state=42)
labels_final = kmeans_final.fit_predict(X)


# 5. GRAFICA 1 – Curva del Silhouette Score

plt.figure(figsize=(8, 5))
plt.plot(resultados_df["k"], resultados_df["Silhouette"], marker="o")
plt.title("Silhouette Score por número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# 6. GRAFICA 2 – Comparación de métricas

plt.figure(figsize=(10, 6))
plt.plot(resultados_df["k"], resultados_df["Calinski-Harabasz"], label="Calinski-Harabasz", marker="o")
plt.plot(resultados_df["k"], resultados_df["Davies-Bouldin"], label="Davies-Bouldin", marker="o")
plt.title("Comparación de métricas de clustering")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Valor de la métrica")
plt.legend()
plt.grid(True)
plt.show()


# 7. GRAFICA 3 — Visualización de clusters en 2D con PCA

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_final, cmap="viridis", s=50)
plt.title(f"Clusters visualizados en PCA 2D (k = {int(mejor_k)})")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

