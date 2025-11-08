
# IMPLEMENTACIÓN INICIAL DEL MODELO SUPERVISADO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# --- Cargar dataset limpio ---
data = pd.read_csv("Morning_Routine_Productivity_CLEAN.csv")

# --- Convertir variable categórica (breakfast_type) a valores numéricos ---
label_encoder = LabelEncoder()
data["breakfast_type"] = label_encoder.fit_transform(data["breakfast_type"])

# ---  Definir variables predictoras (X) y variable objetivo (y) ---
X = data[["sleep_duration_hrs", "meditation_mins", "exercise_mins", "breakfast_type"]]
y = data["productivity_score"]

# ---  Dividir datos en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---  Entrenar modelo ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Hacer predicciones ---
y_pred = model.predict(X_test)

# --- Calcular métricas ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== RESULTADOS DEL MODELO DE REGRESIÓN LINEAL ===")
print(f"MAE  (Error Absoluto Medio): {mae:.3f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.3f}")
print(f"R²   (Coeficiente de Determinación): {r2:.3f}")

# --- Ver coeficientes del modelo ---
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})
print("\nCoeficientes del modelo:")
print(coeficientes)

# --- Comparación con otro modelo: Random Forest Regressor ---
# ---  Entrenar modelo Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ---  Métricas del modelo Random Forest ---
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

# ---  Comparar resultados ---
print("\n=== COMPARACIÓN DE MODELOS ===")
print(f"{'Modelo':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
print(f"{'Regresión Lineal':<20} {mae:<10.3f} {rmse:<10.3f} {r2:<10.3f}")
print(f"{'Random Forest':<20} {rf_mae:<10.3f} {rf_rmse:<10.3f} {rf_r2:<10.3f}")

# ---  Visualización comparativa ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Regresión Lineal')
plt.scatter(y_test, rf_pred, alpha=0.6, label='Random Forest', color='orange')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Comparación de desempeño de modelos')
plt.legend()
plt.grid(True)
plt.show()

