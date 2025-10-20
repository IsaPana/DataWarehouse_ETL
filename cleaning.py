
import pandas as pd
import numpy as np

# Cargar el dataset
file_path = "/Users/erickivannajeracisneros/Downloads/Morning_Routine_Productivity_Dataset_2.csv"
df = pd.read_csv(file_path)

# Revisar estructura general
print("Información inicial:")
print(df.info())
print("\nValores nulos:\n", df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())

#Conversión de tipos de datos
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Wake-up Time'] = pd.to_datetime(df['Wake-up Time'], format='%I:%M %p', errors='coerce').dt.time
df['Work Start Time'] = pd.to_datetime(df['Work Start Time'], format='%I:%M %p', errors='coerce').dt.time

#Estandarización de texto
#Convertir a minúsculas y eliminar espacios en blanco
text_cols = ['Breakfast Type', 'Journaling (Y/N)', 'Mood', 'Notes']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

#Codificación de variables categóricas
# Convierte valores de la columna de si/no a 1/0
#Elimina la columna original
df['Journaling'] = df['Journaling (Y/N)'].map({'yes': 1, 'no': 0})
df.drop(columns=['Journaling (Y/N)'], inplace=True)

#Validación de rangos
# Limitar horas de sueño razonables (3–12)
#Esto para que los datos tengas rangos logicos
df = df[(df['Sleep Duration (hrs)'] >= 3) & (df['Sleep Duration (hrs)'] <= 12)]

# Limitar minutos de ejercicio y meditación
df = df[(df['Exercise (mins)'] >= 0) & (df['Exercise (mins)'] <= 180)]
df = df[(df['Meditation (mins)'] >= 0) & (df['Meditation (mins)'] <= 120)]

#Detección y corrección de outliers (valores que se alejan mucho del rango típico)
# Usaremos el método IQR (rango intercuartílico)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

cols_num = ['Sleep Duration (hrs)', 'Meditation (mins)', 'Exercise (mins)', 'Productivity Score (1-10)']
for col in cols_num:
    df = remove_outliers_iqr(df, col)

#Limpieza básica de texto
df['Notes'] = df['Notes'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

#Renombrar columnas
df.rename(columns={
    'Sleep Duration (hrs)': 'sleep_duration_hrs',
    'Meditation (mins)': 'meditation_mins',
    'Exercise (mins)': 'exercise_mins',
    'Breakfast Type': 'breakfast_type',
    'Work Start Time': 'work_start_time',
    'Wake-up Time': 'wake_up_time',
    'Productivity Score (1-10)': 'productivity_score',
    'Mood': 'mood',
    'Notes': 'notes'
}, inplace=True)

#Exportar dataset limpio
df.to_csv("Morning_Routine_Productivity_CLEAN.csv", index=False)
print("\n Dataset limpio guardado como 'Morning_Routine_Productivity_CLEAN.csv'")
print(f"Total de filas después de limpieza: {len(df)}")

#Vista previa final
print("\nVista previa del dataset limpio:")
print(df.head())
