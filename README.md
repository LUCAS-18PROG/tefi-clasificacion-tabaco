# ClasificaciÃ³n AutomÃ¡tica y Descubrimiento de Patrones de Calidad en Hojas de Tabaco

**InstituciÃ³n:** IES NÂ° 7 "Populorum Progressio - In.Te.La." â€” San Salvador de Jujuy  
**Carrera:** Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial (3.er aÃ±o)  
**CÃ¡tedra:** MinerÃ­a de Datos (2026) â€” Docente: Ing. Fernando Cabana  
**Autor:** Lucas Victor Maciel Alancay Sarapura  

---

## DescripciÃ³n General
Proyecto Final Integrador bajo la **OpciÃ³n D (Dataset Propio)** que articula el Trabajo Especial Final Integrador (TEFI) de visiÃ³n por computadora sobre hojas de tabaco con el ciclo completo **CRISP-DM (Unidades 1 a 6)**:

1. **Unidad 1 (ComprensiÃ³n, EDA y Ã‰tica):** AnÃ¡lisis de 650 registros y 14 variables predictoras (morfologÃ­a, colorimetrÃ­a, textura, deterioro y contexto), criterio de Ã©xito (`Recall >= 0.85`) y protocolo Ã©tico alineado con UNESCO (2021).
2. **Unidad 2 (PreparaciÃ³n sin Fuga de Datos):** EliminaciÃ³n de 3 duplicados (647 registros Ãºnicos), validaciÃ³n de `KNNImputer(k=5)` frente a mediana (reducciÃ³n del 63.3% en MAE), creaciÃ³n de 4 variables agronÃ³micas (`indice_deterioro_total`, `ratio_madurez_cromatico`, `compacidad_foliar`, `flag_moho_critico`) y split estratificado 80/20 previo al escalado.
3. **Unidad 3 (ClasificaciÃ³n):** EvaluaciÃ³n comparativa de Baseline, Ãrbol de DecisiÃ³n, KNN (escalado vs. sin escalar), NaÃ¯ve Bayes, MLP y RegresiÃ³n LogÃ­stica Balanceada.
4. **Unidad 4 (RegresiÃ³n y Complejidad):** PredicciÃ³n del puntaje comercial (0â€“100) en Escenario Pre-curado en campo ($R^2 = 0.9252$) vs. Escenario Post-curado en acopio ($R^2 = 0.9874$), y curva de complejidad (*Underfitting* vs. *Overfitting*).
5. **Unidad 5 (Aprendizaje No Supervisado):** SegmentaciÃ³n de 4 perfiles naturales de cosecha con K-Means y Clustering JerÃ¡rquico Ward ($ARI = 0.9163$), y extracciÃ³n de reglas *Siâ€“Entonces* con Apriori ($Lift = 2.52$).
6. **Unidad 6 (ValidaciÃ³n y Equidad):** ValidaciÃ³n cruzada estratificada de 10 pliegues (`F1 = 0.982 Â± 0.011`, `Recall = 0.976 Â± 0.026`), matriz de confusiÃ³n con 0 falsos negativos y auditorÃ­a de equidad (`Recall = 1.000` en todas las zonas productoras y fuentes Ã³pticas).

---

## InstalaciÃ³n y GeneraciÃ³n del Dataset

```bash
git clone https://github.com/LUCAS-18PROG/tefi-clasificacion-tabaco.git
cd tefi-clasificacion-tabaco
cd tefi-clasificacion-tabaco
pip install -r requirements.txt
python src/generar_dataset.py

