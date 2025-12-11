import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


RUTA_DATA = './muestra'          
RUTA_SALIDA = './submissions' 

ARCHIVO_TRAIN = 'train_1.csv'  
ARCHIVO_KEY = 'key_1.csv'
ARCHIVO_SUBMISSION = 'submission_un_caso_final.csv'

os.makedirs(RUTA_SALIDA, exist_ok=True)

print("--- INICIANDO PROCESO SISTÉMICO (CORRECCIÓN DEFINITIVA DE LÓGICA) ---")


path_train = os.path.join(RUTA_DATA, ARCHIVO_TRAIN)
path_key = os.path.join(RUTA_DATA, ARCHIVO_KEY)

print(f"Cargando archivos CSV desde: {RUTA_DATA}...")

try:
    train_df = pd.read_csv(path_train)
    key_df = pd.read_csv(path_key)
except FileNotFoundError:
    print(f"¡ERROR DE RUTA! Archivos no encontrados.")
    exit()


print(">> Buscando una página con datos en ambos sets (Train y Key)...")

# 1. Obtener una lista de páginas base del archivo KEY (sin la fecha)
# CORRECCIÓN DEFINITIVA: rsplit('_', 1)[0] elimina solo el último '_Fecha'
paginas_en_llave = key_df['Page'].apply(lambda x: x.rsplit('_', 1)[0]).unique()

# 2. Buscar la primera página del TRAIN que coincida
nombre_pagina = None
for page_name_train in train_df['Page']:
    page_base_name = page_name_train
    
    # Comprobamos si el nombre base del train existe en la lista de nombres base de la llave
    if page_base_name in paginas_en_llave:
        nombre_pagina = page_name_train
        break

if nombre_pagina is None:
    # Si este error persiste, significa que las 500 páginas de train y key no se cruzan.
    print("¡ERROR FATAL! No se encontró NINGUNA página en común para el análisis de 'un caso único'.")
    print("La lógica de comparación es correcta. Revisa si los sets de datos de muestra son complementarios.")
    exit()

print(f"\n>> Página seleccionada: {nombre_pagina}")

# Extracción y preprocesamiento de la serie temporal
pagina_objetivo = train_df[train_df['Page'] == nombre_pagina].iloc[0]
serie_temporal = pagina_objetivo.drop('Page')
serie_temporal.index = pd.to_datetime(serie_temporal.index)
serie_temporal = pd.to_numeric(serie_temporal, errors='coerce').fillna(0) 

df_model = pd.DataFrame({'visitas': serie_temporal})
df_model['dias'] = (df_model.index - df_model.index[0]).days

X_train = df_model[['dias']].values
y_train = df_model['visitas'].values

model = LinearRegression()
model.fit(X_train, y_train)

print(">> Generando predicciones para el periodo requerido...")

# Filtramos solo las llaves que corresponden a nuestra página objetivo
llaves_caso = key_df[key_df['Page'].str.startswith(nombre_pagina, na=False)].copy()

# Cálculo de días futuros
llaves_caso['Fecha'] = pd.to_datetime(llaves_caso['Page'].apply(lambda x: x[-10:]))
fecha_inicio = df_model.index[0]
dias_futuros = (llaves_caso['Fecha'] - fecha_inicio).days.values.reshape(-1, 1)

predicciones = model.predict(dias_futuros)
predicciones = np.maximum(predicciones, 0) 

submission = pd.DataFrame({
    'Id': llaves_caso['Id'],
    'Visits': predicciones
})

path_guardado = os.path.join(RUTA_SALIDA, ARCHIVO_SUBMISSION)
submission.to_csv(path_guardado, index=False)

print(f"\n>> ¡PROCESO FINALIZADO CON ÉXITO!")
print(f"Archivo de submission para '{nombre_pagina}' guardado en: {path_guardado}")
print("Muestra de la salida:")
print(submission.head())