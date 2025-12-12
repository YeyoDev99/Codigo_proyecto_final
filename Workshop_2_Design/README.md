# Workshop 2 — System Design (Summary)

## Objective
Build upon the systems analysis from **Workshop 1** to create a robust **system design** that addresses challenges, requirements, and chaotic/sensitive aspects identified earlier.

---

## Key Tasks

### 1. Review of Workshop 1 Findings
- Summarize critical constraints, dataset characteristics, and chaos/sensitivity aspects.
- Ensure the proposed design directly addresses these findings.

### 2. Define System Requirements
- Translate analysis into measurable design requirements (performance, reliability, scalability).
- Consider user needs: interpretability, usability, and security.

### 3. High-Level Architecture
- Propose an architectural blueprint including:
  - **Data ingestion**
  - **Preprocessing/ETL**
  - **Feature engineering**
  - **Model training & validation**
  - **Deployment & monitoring**
- Apply **systems engineering principles** (modularity, scalability, maintainability).

### 4. Sensitivity & Chaos Mitigation
- Strategies to handle sensitive inputs, noisy data, and feedback loops.
- Monitoring, error handling, and stability mechanisms for unpredictable conditions.

### 5. Technical Stack & Implementation
- Recommend tools/frameworks (e.g., Python, scikit-learn, Pytorch).
- Justify choices in terms of scalability, reproducibility, and maintainability.
- Outline integration plan and potential design patterns.

---

## Workshop 2 Development Process Update

The system design was developed following the principles of **Modular Monolith** architecture and incorporating strategies to mitigate the non-linear "**chaos factors**" identified in Workshop 1 (e.g., viral spikes, ambiguous missing data).

### Key Design Outcomes:

* **Architecture:** A **Chain of Responsibility** pattern was implemented for the data flow, ensuring strict sequential execution and data validation between the 8 main modules (Ingestion to Evaluation).
* **Chaos Mitigation:** A **Hierarchical Ensemble** approach was designed, using the **Strategy Pattern** to dynamically select the best base model (ARIMA, Prophet, LSTM) based on the specific characteristics and volatility of each Wikipedia article.
* **Scalability:** The stack relies on **Python** with parallel processing tools like **Joblib** to meet the Non-Functional Requirement (NFR) of processing 145,000 time series efficiently.
* **Design Diagrams:** Comprehensive **Architecture Diagrams** (High-Level Flow, Subsystem Breakdown for Training, and Subsystem Breakdown for Inference/Evaluation) have been created and are referenced in the final PDF to illustrate the system's modularity and iterative feedback loops.