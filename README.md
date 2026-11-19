# Clasificación Automática y Descubrimiento de Patrones de Calidad en Hojas de Tabaco

**Institución:** IES N° 7 "Populorum Progressio - In.Te.La." — San Salvador de Jujuy  
**Carrera:** Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial (3.er año)  
**Cátedra:** Minería de Datos (2026) — Docente: Ing. Fernando Cabana  
**Autor:** Lucas Victor Maciel Alancay Sarapura  

---

## Descripción General
Proyecto Final Integrador bajo la **Opción D (Dataset Propio)** que articula el Trabajo Especial Final Integrador (TEFI) de visión por computadora sobre hojas de tabaco con el ciclo completo **CRISP-DM (Unidades 1 a 6)**:

1. **Unidad 1 (Comprensión, EDA y Ética):** Análisis de 650 registros y 14 variables predictoras (morfología, colorimetría, textura, deterioro y contexto), criterio de éxito (`Recall >= 0.85`) y protocolo ético alineado con UNESCO (2021).
2. **Unidad 2 (Preparación sin Fuga de Datos):** Eliminación de 3 duplicados (647 registros únicos), validación de `KNNImputer(k=5)` frente a mediana (reducción del 63.3% en MAE), creación de 4 variables agronómicas (`indice_deterioro_total`, `ratio_madurez_cromatico`, `compacidad_foliar`, `flag_moho_critico`) y split estratificado 80/20 previo al escalado.
3. **Unidad 3 (Clasificación):** Evaluación comparativa de Baseline, Árbol de Decisión, KNN (escalado vs. sin escalar), Naïve Bayes, MLP y Regresión Logística Balanceada.
4. **Unidad 4 (Regresión y Complejidad):** Predicción del puntaje comercial (0–100) en Escenario Pre-curado en campo ($R^2 = 0.9252$) vs. Escenario Post-curado en acopio ($R^2 = 0.9874$), y curva de complejidad (*Underfitting* vs. *Overfitting*).
5. **Unidad 5 (Aprendizaje No Supervisado):** Segmentación de 4 perfiles naturales de cosecha con K-Means y Clustering Jerárquico Ward ($ARI = 0.9163$), y extracción de reglas *Si–Entonces* con Apriori ($Lift = 2.52$).
6. **Unidad 6 (Validación y Equidad):** Validación cruzada estratificada de 10 pliegues (`F1 = 0.982 ± 0.011`, `Recall = 0.976 ± 0.026`), matriz de confusión con 0 falsos negativos y auditoría de equidad (`Recall = 1.000` en todas las zonas productoras y fuentes ópticas).

---

## Instalación y Generación del Dataset

```bash
git clone https://github.com/LUCAS-18PROG/tefi-clasificacion-tabaco.git
cd tefi-clasificacion-tabaco
cd mineria-datos-clasificacion-tabaco
pip install -r requirements.txt
python src/generar_dataset.py
