# Workshop 3 — Diseño de Sistemas Robusto y Gestión de Proyecto

## Objetivo
El objetivo principal de este workshop fue consolidar el diseño de sistemas propuesto en el **Workshop 2** mediante la integración de **principios de ingeniería robusta**, la definición de una estrategia formal de **gestión de riesgos** y el establecimiento de las **bases de gestión de proyecto** necesarias para guiar el sistema hacia la fase de implementación.

---

## Tareas Clave

### 1. Refinamiento de la Arquitectura Robusta (Abordando NFRs)
- Refinar la arquitectura **Monolito Modular** para asegurar alta **Confiabilidad (NFR3)** y **Mantenibilidad**.
- Detallar la implementación del patrón **Chain of Responsibility (CoR)** para la integridad del flujo de datos.

### 2. Estrategia de Identificación y Mitigación de Riesgos
- Realizar un análisis formal de los puntos de fallo potenciales, incluyendo **Model Drift**, **Fuga de Características (Data Leakage)** y **Sobrecarga Computacional**.
- Proponer estrategias de mitigación concretas (ej. Análisis Post-Predicción Estratificado y Chequeos de Validación Temporal).

### 3. Fundamentos de Gestión de Proyecto
- Definir la metodología de gestión de proyecto (**Híbrido Scrum/Kanban**) adecuada para el equipo de desarrollo.
- Establecer **roles, responsabilidades e hitos**, incluyendo un cronograma simple para rastrear el progreso y las dependencias (Ruta Crítica).

### 4. Mejora Incremental
- Resumir las lecciones aprendidas de los workshops anteriores, mostrando cómo las decisiones de diseño evolucionaron para un sistema más robusto.

---

## Actualización del Proceso de Desarrollo del Workshop 3

La arquitectura del sistema fue endurecida exitosamente enfocándose en la durabilidad y la capacidad de gestión, transformando el diseño teórico en un plan listo para el desarrollo.

### Resultados Clave del Diseño:

* **Refinamiento Arquitectónico:** El **Monolito Modular** ahora aplica explícitamente el patrón **Chain of Responsibility (CoR)** al flujo de datos. Esto asegura que cada dato cumpla con estrictos chequeos de integridad y validación temporal antes de ser procesado, mitigando el riesgo de **Fuga de Características**.
* **Mitigación de Drift:** La amenaza de **Model Drift** se aborda diseñando el módulo de **Análisis Post-Predicción Estratificado (FR4)**. Este módulo desglosa el error (SMAPE) por dimensiones (Lenguaje, Volatilidad) para activar una recalibración **dirigida** del **Hierarchical Ensemble**, en lugar de depender de un reentrenamiento global costoso.
* **Gestión de Proyecto:** Se adoptó una metodología **Híbrida Scrum/Kanban**. Scrum gestiona la implementación de los módulos, mientras que Kanban gestiona el *backlog* continuo de riesgos y mejoras detectadas por el Bucle de Retroalimentación de FR4.
* **Visualización de la Ruta Crítica:** Se definió el cronograma de hitos (**M1** $\to$ **M2** $\to$ **M3** $\to$ **M4**), estableciéndolos como la **Ruta Crítica** para la entrega final del proyecto.

---

## Entregables y Referencia de Archivos

| Nombre del Archivo | Descripción | Referencia en el Reporte (PDF) |
| :--- | :--- | :--- |
| **`Workshop-3.pdf`** | Reporte técnico principal que detalla el refinamiento de la arquitectura, el análisis formal de riesgos y el plan de gestión de proyecto. | Todas las Secciones |
| **`README.md`** | Resumen de los objetivos, tareas y resultados clave del Workshop 3. | N/A |
