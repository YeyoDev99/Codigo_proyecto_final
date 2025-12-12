# Workshop 4 — Simulación del Sistema Kaggle

## Objetivo
El propósito del Workshop 4 fue **validar el comportamiento del sistema diseñado en los Workshops 2 y 3** mediante dos simulaciones complementarias:  
1) una **Simulación Data-Driven** basada en Deep Learning, y  
2) una **Simulación Basada en Eventos** utilizando Autómatas Celulares para modelar la naturaleza caótica del tráfico web.  

Estas simulaciones permiten verificar la robustez de los módulos de entrenamiento, predicción y manejo de caos definidos previamente.

---

## Tareas Clave

### 1. Preparación de Datos y Feature Engineering
- Se utilizó el dataset **train_2.csv** de la competencia *Web Traffic Time Series Forecasting*.  
- Se aplicó la lógica del módulo `extractor.py`, donde un patrón **Regex robusto** extrae metadatos relevantes:  
  *Agent, Site, Country y Term*.  
- Este proceso constituye la base del Módulo 3 (Feature Engineering) definido en la arquitectura.

---

### 2. Simulación Data-Driven (Deep Learning)
- Implementación de un **modelo Encoder–Decoder RNN** con **CudnnGRU** y **Mecanismo de Atención**.  
- El optimizador **COCOB** fue utilizado para eliminar el ajuste manual del learning rate.  
- Se empleó una **pérdida SMAPE diferenciable**, alineada con la métrica oficial del sistema.  
- Los hiperparámetros del set **s32** (definidos en `hparams.py`) permitieron evaluar la estabilidad del modelo durante secuencias largas (≈ 550 días).

**Resultados de esta simulación:**
- El modelo captura correctamente dependencias temporales largas.  
- La atención mejora el enfoque sobre segmentos relevantes de la serie.  
- La arquitectura validó los módulos de Entrenamiento y Predicción definidos en el diseño previo.

---

### 3. Simulación Basada en Eventos (Event-Based)
- Se construyó un **Autómata Celular** para modelar picos virales y comportamientos caóticos.  
- Estados:  
  0. Dormant — comportamiento normal  
  1. Incubating — aumento progresivo  
  2. Viral — explosión caótica  
  3. Recovering — fase de descenso  
- Las reglas incluyeron propagación según vecinos, umbrales estocásticos y transición temporal.

**Hallazgos clave:**
- Los eventos virales generan **clusters** y comportamientos no lineales.  
- Estos resultados validan la inclusión de **rnn stability loss** y **decoder stability loss** en el modelo profundo, evitando sobreajuste a spikes temporales.

---

## Resultados Integrados

| Característica | Simulación Data-Driven | Simulación Event-Based |
|----------------|-----------------------|-------------------------|
| Enfoque | Precisión predictiva (SMAPE) | Dinámica caótica del sistema |
| Valor principal | Captura dependencias largas con atención | Identifica propagación viral y sensibilidad extrema |
| Impacto arquitectónico | Valida módulos de entrenamiento y predicción | Valida mecanismos de control de outliers y estabilidad |

Ambas simulaciones corroboran que el sistema diseñado es **robusto, estable y capaz de adaptarse a escenarios complejos**.

---

## Entregables

| Archivo | Descripción |
|--------|-------------|
| **Workshop-4.pdf** | Reporte completo con metodología, simulaciones y discusión de resultados. |
| **README.md** | Resumen estructurado del Workshop 4. |
| **Workshop_4_Simulation/** | Scripts de extracción, prototipos RNN y modelo de Autómatas Celulares. |

