"""Funciones de apoyo (sin Streamlit) para el panel del TEFI de hojas de tabaco.

Cubre: lectura de anotaciones (formato CSV de Roboflow) e imágenes (ZIP / sueltas / carpeta),
normalización de clases, extracción de características de color y textura por caja,
dibujo de cajas y entrenamiento con división POR FOTO ORIGINAL (sin fuga por aumentación).
"""
import io
import os
import re
import zipfile

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # noqa: F401  (disponible por si se amplía el pipeline)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

COLUMNAS_ANOTACION = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
EXT_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

ORDEN_CLASES = ["Sana", "No sana", "Sin etiqueta"]
COLOR_CLASE = {"Sana": "#2E8B57", "No sana": "#C0392B", "Sin etiqueta": "#7F8C8D"}
COLOR_CLASE_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in COLOR_CLASE.items()}

# Características que se extraen de la imagen (recorte de cada caja)
FEATURES_COLOR = ["R_medio", "G_medio", "B_medio", "indice_amarillamiento", "brillo", "saturacion",
                  "textura_laplaciano", "pct_oscuro", "pct_amarillo", "pct_verde", "pct_marron"]
FEATURES_GEOMETRIA = ["pct_area_imagen", "relacion_aspecto"]
ETIQUETAS_FEATURES = {
    "R_medio": "Rojo medio (R)", "G_medio": "Verde medio (G)", "B_medio": "Azul medio (B)",
    "indice_amarillamiento": "Índice de amarillamiento", "brillo": "Brillo (V)", "saturacion": "Saturación (S)",
    "textura_laplaciano": "Textura (log varianza Laplaciano)", "pct_oscuro": "% píxeles oscuros",
    "pct_amarillo": "% píxeles amarillos", "pct_verde": "% píxeles verdes", "pct_marron": "% píxeles marrones",
    "pct_area_imagen": "% del área de la imagen", "relacion_aspecto": "Relación de aspecto (ancho/alto)",
}


# ---------------------------------------------------------------------------------------------
# Anotaciones
# ---------------------------------------------------------------------------------------------
def normalizar_clase(valor):
    """Unifica los nombres de clase (p. ej. 'Unhealthy' y 'unhealthy tobacco leaf' -> 'No sana')."""
    t = str(valor).strip().lower()
    if t in ("", "nan", "none") or "unlabeled" in t:
        return "Sin etiqueta"
    if "unhealthy" in t or "enferm" in t or "no sana" in t or "diseas" in t:
        return "No sana"
    if "leaf" in t or "tobacco" in t or "healthy" in t or "sana" in t or "hoja" in t:
        return "Sana"
    return str(valor).strip().capitalize()


def extraer_fuente(nombre):
    """Identifica la FOTO ORIGINAL de un archivo exportado por Roboflow.

    'XXXX_n_jpg.rf.<hash>.jpg' -> 'XXXX_n'. Todas las versiones aumentadas (flips) de una misma foto
    comparten esta clave; sirve para dividir train/test sin fuga de información.
    """
    base = os.path.basename(str(nombre))
    sin_rf = re.sub(r"_(jpe?g|png)\.rf\.[0-9a-fA-F]+(\.\w+)?$", "", base, flags=re.I)
    return sin_rf if sin_rf != base else os.path.splitext(base)[0]


def preparar_anotaciones(df_bruto, particion=None):
    """Valida el CSV, normaliza clases y agrega columnas geométricas derivadas."""
    faltan = [c for c in COLUMNAS_ANOTACION if c not in df_bruto.columns]
    if faltan:
        raise ValueError("El CSV no tiene el formato esperado (Roboflow, TensorFlow Object Detection). "
                         f"Faltan las columnas: {', '.join(faltan)}.")
    d = df_bruto[COLUMNAS_ANOTACION].copy()
    for c in ["width", "height", "xmin", "ymin", "xmax", "ymax"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["filename", "width", "height", "xmin", "ymin", "xmax", "ymax"]).copy()
    d["clase_original"] = d["class"].astype(str)
    d["clase"] = d["class"].map(normalizar_clase)
    d["fuente"] = d["filename"].map(extraer_fuente)
    d["ancho_px"] = d["xmax"] - d["xmin"]
    d["alto_px"] = d["ymax"] - d["ymin"]
    d["area_caja_px"] = d["ancho_px"] * d["alto_px"]
    d["pct_area_imagen"] = 100 * d["area_caja_px"] / (d["width"] * d["height"])
    d["relacion_aspecto"] = d["ancho_px"] / d["alto_px"].replace(0, np.nan)
    d["cx"] = (d["xmin"] + d["xmax"]) / 2
    d["cy"] = (d["ymin"] + d["ymax"]) / 2
    d["toca_borde"] = ((d["xmin"] <= 1) | (d["ymin"] <= 1) |
                       (d["xmax"] >= d["width"] - 1) | (d["ymax"] >= d["height"] - 1))
    if particion is not None:
        d["particion"] = particion
    return d.drop(columns=["class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Lectura de imágenes (ZIP / carpeta)
# ---------------------------------------------------------------------------------------------
def leer_zip(datos_zip):
    """Devuelve ({nombre_archivo: bytes}, [(particion, bytes_csv), ...]) desde un ZIP (p. ej. export de Roboflow)."""
    imagenes, csvs = {}, []
    with zipfile.ZipFile(io.BytesIO(datos_zip)) as z:
        for info in z.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            base = os.path.basename(info.filename)
            if base.lower().endswith(EXT_IMAGEN):
                imagenes[base] = z.read(info)
            elif base.lower() == "_annotations.csv":
                particion = os.path.basename(os.path.dirname(info.filename)) or "(raíz)"
                csvs.append((particion, z.read(info)))
    return imagenes, csvs


def leer_carpeta(ruta):
    """Lee imágenes (y _annotations.csv, si hay) de una carpeta del disco, incluyendo subcarpetas."""
    ruta = os.path.expanduser(str(ruta).strip())
    if not os.path.isdir(ruta):
        raise ValueError(f"La carpeta no existe: {ruta}")
    imagenes, csvs = {}, []
    for raiz, _, archivos in os.walk(ruta):
        for nombre in archivos:
            completo = os.path.join(raiz, nombre)
            if nombre.lower().endswith(EXT_IMAGEN):
                with open(completo, "rb") as f:
                    imagenes[nombre] = f.read()
            elif nombre.lower() == "_annotations.csv":
                with open(completo, "rb") as f:
                    csvs.append((os.path.basename(raiz) or "(raíz)", f.read()))
    return imagenes, csvs


def decodificar_imagen(datos):
    """bytes -> arreglo RGB (alto, ancho, 3)."""
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def emparejar_imagenes(df, imagenes):
    """Para cada nombre del CSV devuelve la clave en `imagenes` (coincidencia exacta o sin distinguir mayúsculas)."""
    minus = {k.lower(): k for k in imagenes}
    mapa = {}
    for nombre in df["filename"].unique():
        if nombre in imagenes:
            mapa[nombre] = nombre
        else:
            mapa[nombre] = minus.get(str(nombre).lower())
    return mapa


def dibujar_cajas(img_rgb, cajas):
    """Dibuja las cajas (escalando si la imagen real no mide lo que dice el CSV)."""
    salida = img_rgb.copy()
    h, w = salida.shape[:2]
    for _, r in cajas.iterrows():
        sx, sy = w / r["width"], h / r["height"]
        p1 = (int(r["xmin"] * sx), int(r["ymin"] * sy))
        p2 = (int(r["xmax"] * sx), int(r["ymax"] * sy))
        color = COLOR_CLASE_RGB.get(r["clase"], (52, 152, 219))
        cv2.rectangle(salida, p1, p2, color, 2)
        cv2.putText(salida, str(r["clase"]), (p1[0] + 2, max(14, p1[1] + 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return salida


# ---------------------------------------------------------------------------------------------
# Características de color y textura por caja
# ---------------------------------------------------------------------------------------------
def _caracteristicas_region(region_rgb, excluir_fondo, umbral_sat):
    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    mascara = np.ones(gris.shape, dtype=bool)
    if excluir_fondo:
        m = hsv[..., 1] >= umbral_sat                      # el fondo (blanco/gris) tiene poca saturación
        if m.mean() >= 0.05:                               # si la máscara es absurda, se ignora
            mascara = m
    px = region_rgb[mascara].astype(float)
    R, G, B = px.mean(axis=0)
    H = hsv[..., 0][mascara].astype(float)
    S = hsv[..., 1][mascara].astype(float)
    V = hsv[..., 2][mascara].astype(float)
    lap = cv2.Laplacian(gris, cv2.CV_64F)[mascara]
    return {
        "R_medio": R, "G_medio": G, "B_medio": B,
        "indice_amarillamiento": float(np.clip((G / max(R, 1.0) - 0.5) / 0.8, 0, 1)),   # misma fórmula que el notebook
        "brillo": V.mean(), "saturacion": S.mean(),
        "textura_laplaciano": float(np.log1p(lap.var())),
        "pct_oscuro": 100 * float((V < 60).mean()),
        "pct_amarillo": 100 * float(((H >= 20) & (H <= 35) & (S >= 60)).mean()),
        "pct_verde": 100 * float(((H >= 36) & (H <= 85) & (S >= 40)).mean()),
        "pct_marron": 100 * float(((H >= 5) & (H < 20) & (S >= 50) & (V >= 40) & (V <= 200)).mean()),
    }


def caracteristicas_cajas(df, imagenes, excluir_fondo=False, umbral_sat=40, recorte_pct=10):
    """Calcula las características de cada caja cuya imagen esté disponible. Devuelve DataFrame con el índice de `df`."""
    mapa = emparejar_imagenes(df, imagenes)
    filas = {}
    for nombre, grupo in df.groupby("filename", sort=False):
        clave = mapa.get(nombre)
        if clave is None:
            continue
        try:
            img = decodificar_imagen(imagenes[clave])
        except Exception:
            continue
        h, w = img.shape[:2]
        for idx, r in grupo.iterrows():
            sx, sy = w / r["width"], h / r["height"]
            x1, x2, y1, y2 = r["xmin"] * sx, r["xmax"] * sx, r["ymin"] * sy, r["ymax"] * sy
            dx, dy = (x2 - x1) * recorte_pct / 100, (y2 - y1) * recorte_pct / 100
            x1, x2, y1, y2 = int(max(0, x1 + dx)), int(min(w, x2 - dx)), int(max(0, y1 + dy)), int(min(h, y2 - dy))
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue                                     # caja demasiado chica para medir color/textura
            filas[idx] = _caracteristicas_region(img[y1:y2, x1:x2], excluir_fondo, umbral_sat)
    if not filas:
        return pd.DataFrame(columns=FEATURES_COLOR)
    return pd.DataFrame.from_dict(filas, orient="index")[FEATURES_COLOR]


# ---------------------------------------------------------------------------------------------
# Entrenamiento (clasificación Sana vs No sana) con división por foto original
# ---------------------------------------------------------------------------------------------
NOMBRES_MODELOS = ["Random Forest", "Gradient Boosting", "SVM (RBF)", "Regresión Logística"]


def construir_modelo(nombre, semilla=42):
    if nombre == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                     random_state=semilla, n_jobs=-1)
    elif nombre == "Gradient Boosting":
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200,
                                             class_weight="balanced", random_state=semilla)
    elif nombre == "SVM (RBF)":
        clf = SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced", probability=True, random_state=semilla)
    elif nombre == "Regresión Logística":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000)
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return Pipeline([("imputador", SimpleImputer(strategy="median")), ("escalador", RobustScaler()), ("clf", clf)])


def metricas_binarias(y, proba, umbral=0.5):
    """Métricas para la clase positiva (1 = No sana) a un umbral dado + matriz de confusión [[tn, fp], [fn, tp]]."""
    y = np.asarray(y)
    pred = (np.asarray(proba) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "Accuracy": accuracy_score(y, pred),
        "Accuracy balanceada": balanced_accuracy_score(y, pred),
        "Precisión (No sana)": precision_score(y, pred, zero_division=0),
        "Recall (No sana)": recall_score(y, pred, zero_division=0),
        "F1 (No sana)": f1_score(y, pred, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        out["ROC-AUC"] = roc_auc_score(y, proba)
        out["PR-AUC"] = average_precision_score(y, proba)
    return out, np.array([[tn, fp], [fn, tp]])


def entrenar_evaluar(df_feat, columnas, modelos, tam_test=0.25, semilla=42, k_cv=5):
    """Entrena y evalúa agrupando por foto original (StratifiedGroupKFold): ninguna foto queda en train y test a la vez."""
    d = df_feat[df_feat["clase"].isin(["Sana", "No sana"])].dropna(subset=["fuente"]).copy()
    if d.empty or d["clase"].nunique() < 2:
        raise ValueError("Se necesitan hojas 'Sana' y 'No sana' con características calculadas.")
    y = (d["clase"] == "No sana").astype(int).to_numpy()
    grupos = d["fuente"].to_numpy()
    X = d[columnas]

    n_pos_grupos = len(set(grupos[y == 1]))
    if n_pos_grupos < 2:
        raise ValueError("Hay menos de 2 fotos originales con hojas no sanas: no se puede separar entrenamiento y prueba.")

    # --- partición hold-out: se toma el primer fold válido de un StratifiedGroupKFold ---
    n_splits = int(min(max(2, round(1 / tam_test)), n_pos_grupos))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    idx_tr = idx_te = None
    for tr, te in sgkf.split(X, y, grupos):
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            idx_tr, idx_te = tr, te
            break
    if idx_tr is None:
        raise ValueError("No se pudo armar una partición con ambas clases en entrenamiento y prueba.")
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]

    resultado = {"modelos": {}, "y_test": y_te, "X_test": X_te,
                 "n_train": len(idx_tr), "n_test": len(idx_te),
                 "pos_train": int(y_tr.sum()), "pos_test": int(y_te.sum()),
                 "fotos_train": len(set(grupos[idx_tr])), "fotos_test": len(set(grupos[idx_te])),
                 "fotos_compartidas": len(set(grupos[idx_tr]) & set(grupos[idx_te])),
                 "columnas": list(columnas)}
    for nombre in modelos:
        pipe = construir_modelo(nombre, semilla)
        pipe.fit(X_tr, y_tr)
        resultado["modelos"][nombre] = {"pipe": pipe, "proba": pipe.predict_proba(X_te)[:, 1]}

    # --- validación cruzada agrupada sobre todos los datos ---
    k = int(min(k_cv, n_pos_grupos))
    filas_cv = []
    if k >= 2:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
        scoring = {"Accuracy balanceada": "balanced_accuracy", "F1 (No sana)": "f1",
                   "ROC-AUC": "roc_auc", "PR-AUC": "average_precision"}
        for nombre in modelos:
            r = cross_validate(construir_modelo(nombre, semilla), X, y, groups=grupos, cv=cv,
                               scoring=scoring, error_score=np.nan)
            for i in range(k):
                fila = {"Modelo": nombre, "Fold": i + 1}
                fila.update({m: r[f"test_{m}"][i] for m in scoring})
                filas_cv.append(fila)
    resultado["cv"] = pd.DataFrame(filas_cv) if filas_cv else None
    resultado["k_cv"] = k

    # --- mejor modelo (PR-AUC en hold-out) e importancia por permutación ---
    pr = {n: average_precision_score(y_te, m["proba"]) for n, m in resultado["modelos"].items()}
    mejor = max(pr, key=pr.get)
    resultado["mejor"] = mejor
    try:
        perm = permutation_importance(resultado["modelos"][mejor]["pipe"], X_te, y_te, scoring="average_precision",
                                      n_repeats=15, random_state=semilla)
        resultado["importancias"] = pd.DataFrame({"variable": list(columnas), "importancia": perm.importances_mean,
                                                  "desvio": perm.importances_std}).sort_values("importancia")
    except Exception:
        resultado["importancias"] = None
    return resultado
