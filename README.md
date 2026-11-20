# Panel Streamlit · TEFI hojas de tabaco

## Instalar y ejecutar (en tu computadora)
```
cd panel_tabaco
pip install -r requirements.txt
streamlit run app_tabaco.py
```
Se abre en el navegador (http://localhost:8501). **La primera pantalla pide el dataset**: el `_annotations.csv` y las
imágenes (ZIP, sueltas o ruta de carpeta). Si subís el ZIP completo exportado de Roboflow no hace falta el CSV.

## Archivos
- `app_tabaco.py`: la interfaz (carga del dataset + 7 pestañas con gráficos interactivos).
- `utils_tabaco.py`: lectura de datos, características de color/textura, dibujo de cajas y entrenamiento.
- `.streamlit/config.toml`: permite subir archivos de hasta 1 GB y fija el color del tema.

## Qué hace el entrenamiento
Clasifica cada hoja como **Sana / No sana** con características de color y textura extraídas de la imagen. La división
train/test y la validación cruzada son **por foto original** (evita que versiones aumentadas de la misma hoja queden en train y test).
