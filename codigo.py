import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ---------------------------------------------------------
# Nota: Siguiendo tus instrucciones, el código referencia los nombres de la Etapa 1,
# pero internamente carga tus muestras '.txt' de la Etapa 2 para que funcione aquí.
ARCHIVO_ENTRENAMIENTO = 'train_1.csv'  # En la demo usamos: 'train_2_500.txt'
ARCHIVO_LLAVE = 'key_1.csv'            # En la demo usamos: 'key_2_500.txt'

print("Cargando datos...")

# Intentamos cargar los archivos. Si estás ejecutando esto en tu local,
# asegúrate de que los nombres coincidan. Aquí uso los que subiste.
try:
    # Leemos los txt como si fueran los csv completos
    train_df = pd.read_csv('./muestra/train_2_500.txt')
    key_df = pd.read_csv('./muestra/key_2_500.txt')
except FileNotFoundError:
    print("No se encontraron los archivos de muestra. Asegúrate de tener los CSV.")
    # Datos dummy por si acaso fallara la carga en otro entorno
    train_df = pd.DataFrame() 
    key_df = pd.DataFrame()

# ---------------------------------------------------------
# 2. SELECCIÓN DE UN SOLO CASO (PÁGINA)
# ---------------------------------------------------------
# Seleccionamos la primera página del set de entrenamiento para el ejemplo.
# En el archivo real hay 145k páginas.
if not train_df.empty:
    nombre_pagina_objetivo = train_df['Page'].iloc[0]
    print(f"Procesando página: {nombre_pagina_objetivo}")

    # Extraemos la serie de tiempo para esta página
    # Quitamos la columna 'Page' para tener solo fechas y visitas
    serie_entrenamiento = train_df[train_df['Page'] == nombre_pagina_objetivo].drop('Page', axis=1).T
    serie_entrenamiento.columns = ['Visitas']
    # Convertimos el índice a formato fecha
    serie_entrenamiento.index = pd.to_datetime(serie_entrenamiento.index)

    # ---------------------------------------------------------
    # 3. PREPROCESAMIENTO (Módulo del Sistema)
    # ---------------------------------------------------------
    # Manejo de valores nulos (NaN): Los llenamos con 0 como sugiere la competencia
    serie_entrenamiento['Visitas'] = serie_entrenamiento['Visitas'].fillna(0)

    # Ingeniería de Características (Feature Engineering)
    # Creamos una variable numérica 'Dias' para que la regresión entienda el tiempo
    serie_entrenamiento['Dias'] = (serie_entrenamiento.index - serie_entrenamiento.index[0]).days
    
    X_train = serie_entrenamiento[['Dias']].values
    y_train = serie_entrenamiento['Visitas'].values

    # ---------------------------------------------------------
    # 4. MODELADO (Simulación del Meta-Modelo)
    # ---------------------------------------------------------
    # Usamos Regresión Lineal para capturar la tendencia básica.
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. PREDICCIÓN (Forecasting)
    # ---------------------------------------------------------
    # Necesitamos predecir para las fechas que pide el archivo 'key'.
    # El archivo key tiene el formato: NombrePagina_Fecha
    
    # Filtramos las llaves que corresponden a NUESTRA página objetivo
    llaves_pagina = key_df[key_df['Page'].str.contains(nombre_pagina_objetivo, regex=False)].copy()

    if llaves_pagina.empty:
        print("Advertencia: No se encontraron llaves para esta página en el archivo de muestra.")
    else:
        # Extraer la fecha del string largo. (Los últimos 10 caracteres son YYYY-MM-DD)
        llaves_pagina['Fecha'] = pd.to_datetime(llaves_pagina['Page'].apply(lambda x: x[-10:]))
        fechas_futuras = llaves_pagina['Fecha']

        # Preparamos los días futuros numéricamente (basado en el inicio de la serie)
        dias_inicio = serie_entrenamiento.index[0]
        dias_futuros = (fechas_futuras - dias_inicio).days.values.reshape(-1, 1)

        # Hacemos la predicción
        predicciones = modelo.predict(dias_futuros)
        
        # Corrección: El tráfico no puede ser negativo
        predicciones = np.maximum(predicciones, 0)

        # ---------------------------------------------------------
        # 6. GENERACIÓN DEL ARCHIVO DE ENVÍO (SUBMISSION)
        # ---------------------------------------------------------
        # Creamos el dataframe final con Id y Visitas
        submission = pd.DataFrame({
            'Id': llaves_pagina['Id'],
            'Visits': predicciones
        })

        print("Predicción completada exitosamente para el caso.")
        print(submission.head())
        
        # Guardar resultado (opcional)
        # submission.to_csv('submission_1.csv', index=False)

else:
    print("El dataframe de entrenamiento está vacío.")