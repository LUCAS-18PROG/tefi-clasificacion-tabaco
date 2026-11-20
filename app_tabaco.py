"""Panel Streamlit · TEFI: Hojas de tabaco (dataset Roboflow de detección + entrenamiento).

Ejecutar:  streamlit run app_tabaco.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

import utils_tabaco as ut

st.set_page_config(page_title="TEFI · Hojas de tabaco", page_icon="🍃", layout="wide")

COLORES = ut.COLOR_CLASE
ORDEN = {"clase": ut.ORDEN_CLASES}
ETQ = ut.ETIQUETAS_FEATURES


def grafico(fig, clave, alto=420):
    """Aplica un estilo común y dibuja el gráfico (cada gráfico lleva una clave única)."""
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, key=clave)


# =============================================================================================
# PASO 1 · CARGA DEL DATASET (antes de mostrar cualquier otra cosa)
# =============================================================================================
def pantalla_carga():
    st.title("🍃 Hojas de tabaco · Panel de datos y entrenamiento")
    st.caption("Trabajo Especial Final Integrador · Minería de Datos · CRISP-DM")
    st.subheader("Paso 1 · Cargá tu dataset")
    st.write("Subí el archivo de anotaciones y las imágenes. Nada se analiza hasta que presiones **Cargar dataset**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**① Anotaciones (CSV de Roboflow)**")
        csv_sub = st.file_uploader("Archivo `_annotations.csv`", type=["csv"], key="up_csv")
        st.caption("Columnas esperadas: filename, width, height, class, xmin, ymin, xmax, ymax. "
                   "Si subís el ZIP completo de Roboflow no hace falta este archivo.")
    with col2:
        st.markdown("**② Imágenes de entrenamiento**")
        modo = st.radio("¿Cómo querés cargar las imágenes?",
                        ["ZIP con las imágenes", "Imágenes sueltas", "Ruta de una carpeta (la app corre en tu PC)",
                         "Todavía no (solo el CSV)"], key="modo_img")
        zip_sub, sueltas, ruta = None, None, ""
        if modo.startswith("ZIP"):
            zip_sub = st.file_uploader("ZIP con las imágenes (o el export completo de Roboflow)", type=["zip"], key="up_zip")
        elif modo.startswith("Imágenes"):
            sueltas = st.file_uploader("Imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_imgs")
        elif modo.startswith("Ruta"):
            ruta = st.text_input("Ruta de la carpeta", placeholder="/home/usuario/tabaco/train", key="up_ruta")

    if st.button("Cargar dataset", type="primary"):
        imagenes, csvs_zip = {}, []
        try:
            if zip_sub is not None:
                imagenes, csvs_zip = ut.leer_zip(zip_sub.getvalue())
            elif sueltas:
                imagenes = {os.path.basename(f.name): f.getvalue() for f in sueltas}
            elif ruta.strip():
                imagenes, csvs_zip = ut.leer_carpeta(ruta)

            tablas = []
            if csv_sub is not None:
                tablas.append(ut.preparar_anotaciones(pd.read_csv(io.BytesIO(csv_sub.getvalue())), particion="(subido)"))
            else:
                for particion, contenido in csvs_zip:
                    tablas.append(ut.preparar_anotaciones(pd.read_csv(io.BytesIO(contenido)), particion=particion))
            if not tablas:
                st.error("Falta el archivo de anotaciones: subí el `_annotations.csv` o un ZIP que lo incluya.")
                return
            df = pd.concat(tablas, ignore_index=True)
            if df.empty:
                st.error("El CSV no tiene filas válidas.")
                return
            clave = f"{len(df)}-{len(imagenes)}-{sum(len(b) for b in imagenes.values())}"
            st.session_state["datos"] = {"df": df, "imagenes": imagenes, "clave": clave}
        except Exception as e:                                   # mensajes claros en vez de un traceback
            st.error(f"No se pudo cargar el dataset: {e}")
        if "datos" in st.session_state:
            st.rerun()


datos = st.session_state.get("datos")
if datos is None:
    pantalla_carga()
    st.stop()

df = datos["df"]
imagenes = datos["imagenes"]
mapa = ut.emparejar_imagenes(df, imagenes)
n_emparejadas = sum(1 for v in mapa.values() if v is not None)
hay_imagenes = n_emparejadas > 0

# =============================================================================================
# BARRA LATERAL
# =============================================================================================
with st.sidebar:
    st.header("🍃 Dataset cargado")
    st.write(f"**{df['filename'].nunique()}** imágenes anotadas · **{len(df)}** cajas")
    st.write(f"Imágenes encontradas: **{n_emparejadas}** de {len(mapa)}")
    if st.button("🔄 Cargar otro dataset"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.subheader("Filtros")
    clases_disp = [c for c in ut.ORDEN_CLASES if c in set(df["clase"])] + \
                  [c for c in sorted(set(df["clase"])) if c not in ut.ORDEN_CLASES]
    clases_sel = st.multiselect("Clases a mostrar", clases_disp, default=clases_disp)
    if "particion" in df.columns and df["particion"].nunique() > 1:
        parts = sorted(df["particion"].unique())
        parts_sel = st.multiselect("Partición", parts, default=parts)
    else:
        parts_sel = None
    st.divider()
    st.subheader("Extracción de características")
    excluir_fondo = st.checkbox("Excluir fondo (píxeles poco saturados)", value=False)
    umbral_sat = st.slider("Umbral de saturación", 0, 120, 40, disabled=not excluir_fondo)
    recorte = st.slider("Recorte de bordes de la caja (%)", 0, 40, 10,
                        help="Descarta el borde de cada caja, donde suele haber fondo.")

dff = df[df["clase"].isin(clases_sel)]
if parts_sel is not None:
    dff = dff[dff["particion"].isin(parts_sel)]


@st.cache_data(show_spinner="Extrayendo características de las imágenes…")
def calcular_features(df_cajas, _imagenes, clave, excluir, umbral, recorte_pct):
    return ut.caracteristicas_cajas(df_cajas, _imagenes, excluir, umbral, recorte_pct)


def obtener_features():
    if not hay_imagenes:
        return None
    feats = calcular_features(df, imagenes, datos["clave"], excluir_fondo, umbral_sat, recorte)
    if feats.empty:
        return None
    sal = df.join(feats, how="inner")
    sal = sal[sal["clase"].isin(clases_sel)]
    if parts_sel is not None:
        sal = sal[sal["particion"].isin(parts_sel)]
    return sal


st.title("🍃 Hojas de tabaco · Panel de datos y entrenamiento")
if dff.empty:
    st.warning("Los filtros dejaron el dataset vacío. Ajustá la selección en la barra lateral.")
    st.stop()

tab_res, tab_cls, tab_geo, tab_gal, tab_col, tab_ent, tab_cal = st.tabs(
    ["📋 Resumen", "📊 Clases", "📐 Geometría", "🖼️ Galería", "🎨 Color y textura", "🤖 Entrenamiento", "✅ Calidad del dataset"])

# =============================================================================================
# 📋 RESUMEN
# =============================================================================================
with tab_res:
    n_imgs, n_cajas, n_fotos = dff["filename"].nunique(), len(dff), dff["fuente"].nunique()
    n_ns = int((dff["clase"] == "No sana").sum())
    c = st.columns(5)
    c[0].metric("Imágenes anotadas", n_imgs)
    c[1].metric("Hojas anotadas (cajas)", n_cajas)
    c[2].metric("Fotos originales", n_fotos)
    c[3].metric("Hojas no sanas", n_ns, f"{100 * n_ns / max(n_cajas, 1):.1f} % del total", delta_color="off")
    c[4].metric("Imágenes cargadas", f"{n_emparejadas}/{len(mapa)}")

    versiones = dff.groupby("fuente")["filename"].nunique()
    st.subheader("Lectura rápida")
    if n_ns and n_cajas:
        st.info(f"**Desbalance de clases:** solo {100 * n_ns / n_cajas:.1f} % de las hojas anotadas está marcado como "
                f"*No sana*. Al evaluar usá F1, PR-AUC o accuracy balanceada, no la accuracy simple.")
    if versiones.max() > 1:
        st.warning(f"**Aumentación de datos:** las {n_imgs} imágenes provienen de solo {n_fotos} fotos originales "
                   f"(hasta {int(versiones.max())} versiones por foto). Si dividís train/test al azar, la misma hoja "
                   f"aparecería en ambos lados y las métricas saldrían infladas. El entrenamiento de este panel "
                   f"divide **por foto original**.")
    mapeo = (df.groupby(["clase_original", "clase"]).size().reset_index(name="cajas")
               .rename(columns={"clase_original": "Nombre en el CSV", "clase": "Clase unificada"}))
    st.markdown("**Unificación de clases** (verificá que el mapeo sea correcto):")
    st.dataframe(mapeo)
    st.markdown("**Vista previa de las anotaciones**")
    st.dataframe(dff.head(50))

# =============================================================================================
# 📊 CLASES
# =============================================================================================
with tab_cls:
    cnt = dff["clase"].value_counts().reindex([c for c in ut.ORDEN_CLASES if c in set(dff["clase"])] +
                                              [c for c in dff["clase"].unique() if c not in ut.ORDEN_CLASES]).dropna()
    cnt = cnt.rename_axis("clase").reset_index(name="cajas")
    a, b = st.columns(2)
    with a:
        fig = px.bar(cnt, x="clase", y="cajas", color="clase", text="cajas", color_discrete_map=COLORES,
                     title="Cantidad de hojas por clase")
        fig.update_layout(showlegend=False, xaxis_title="Clase", yaxis_title="Cantidad de hojas")
        grafico(fig, "g_cls_bar")
    with b:
        fig = px.pie(cnt, names="clase", values="cajas", hole=0.55, color="clase", color_discrete_map=COLORES,
                     title="Proporción de cada clase")
        grafico(fig, "g_cls_pie")

    a, b = st.columns(2)
    with a:
        por_img = dff.groupby("filename").size().reset_index(name="cajas_por_imagen")
        fig = px.histogram(por_img, x="cajas_por_imagen", nbins=int(max(por_img["cajas_por_imagen"].max(), 2)),
                           title="Hojas anotadas por imagen", color_discrete_sequence=["#4C78A8"])
        fig.update_layout(xaxis_title="Hojas por imagen", yaxis_title="Cantidad de imágenes")
        grafico(fig, "g_cls_porimg")
    with b:
        ver = dff.groupby("fuente")["filename"].nunique().reset_index(name="versiones")
        fig = px.histogram(ver, x="versiones", title="Versiones (aumentación) por foto original",
                           color_discrete_sequence=["#E8803A"])
        fig.update_layout(xaxis_title="Versiones de la misma foto", yaxis_title="Cantidad de fotos originales",
                          xaxis=dict(dtick=1))
        grafico(fig, "g_cls_versiones")

    pos_por_foto = (dff.assign(es_no_sana=dff["clase"] == "No sana").groupby("fuente")["es_no_sana"].any()
                    .map({True: "Con hojas no sanas", False: "Solo hojas sanas"}).value_counts().reset_index())
    pos_por_foto.columns = ["tipo_de_foto", "fotos"]
    fig = px.bar(pos_por_foto, x="tipo_de_foto", y="fotos", text="fotos", color="tipo_de_foto",
                 color_discrete_sequence=["#C0392B", "#2E8B57"], title="Fotos originales según si tienen hojas no sanas")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Fotos originales")
    grafico(fig, "g_cls_fotos", alto=360)

    if "particion" in dff.columns and dff["particion"].nunique() > 1:
        tabla = dff.groupby(["particion", "clase"]).size().reset_index(name="cajas")
        fig = px.bar(tabla, x="particion", y="cajas", color="clase", barmode="stack", color_discrete_map=COLORES,
                     title="Clases por partición del dataset")
        fig.update_layout(xaxis_title="Partición", yaxis_title="Cajas")
        grafico(fig, "g_cls_part")

# =============================================================================================
# 📐 GEOMETRÍA
# =============================================================================================
with tab_geo:
    a, b = st.columns(2)
    with a:
        fig = px.histogram(dff, x="pct_area_imagen", color="clase", nbins=40, barmode="overlay", opacity=0.7,
                           marginal="box", color_discrete_map=COLORES, category_orders=ORDEN,
                           title="Tamaño de la hoja respecto de la imagen")
        fig.update_layout(xaxis_title="% del área de la imagen", yaxis_title="Cantidad de hojas")
        grafico(fig, "g_geo_hist", alto=470)
    with b:
        fig = px.scatter(dff, x="ancho_px", y="alto_px", color="clase", opacity=0.65, color_discrete_map=COLORES,
                         category_orders=ORDEN, hover_data=["filename"], title="Ancho vs. alto de cada caja")
        fig.update_layout(xaxis_title="Ancho (px)", yaxis_title="Alto (px)")
        grafico(fig, "g_geo_scatter", alto=470)

    a, b = st.columns(2)
    with a:
        fig = px.density_heatmap(dff, x="cx", y="cy", nbinsx=24, nbinsy=24, color_continuous_scale="YlGn",
                                 title="¿Dónde aparecen las hojas en la imagen? (centro de cada caja)")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Posición horizontal (px)", yaxis_title="Posición vertical (px)",
                          coloraxis_colorbar_title="Cajas")
        grafico(fig, "g_geo_heat")
    with b:
        fig = px.box(dff, x="clase", y="relacion_aspecto", color="clase", points="outliers",
                     color_discrete_map=COLORES, category_orders=ORDEN, title="Relación de aspecto por clase")
        fig.update_layout(showlegend=False, xaxis_title="Clase", yaxis_title="Ancho / alto")
        grafico(fig, "g_geo_aspecto")

    borde = (dff.groupby("clase")["toca_borde"].mean() * 100).reset_index(name="pct")
    fig = px.bar(borde, x="clase", y="pct", color="clase", text=borde["pct"].round(1), color_discrete_map=COLORES,
                 title="% de cajas que tocan el borde de la imagen (hoja posiblemente cortada)")
    fig.update_layout(showlegend=False, xaxis_title="Clase", yaxis_title="% de cajas")
    grafico(fig, "g_geo_borde", alto=360)

# =============================================================================================
# 🖼️ GALERÍA
# =============================================================================================
with tab_gal:
    if not hay_imagenes:
        st.info("Cargá las imágenes en el Paso 1 (botón «Cargar otro dataset») para ver la galería.")
    else:
        a, b, c3 = st.columns(3)
        solo_ns = a.checkbox("Solo imágenes con hojas no sanas", value=True)
        n_mostrar = b.slider("Cantidad de imágenes", 3, 24, 9)
        semilla_g = c3.number_input("Semilla del muestreo", 0, 9999, 42)
        base = dff[dff["clase"] == "No sana"] if solo_ns else dff
        nombres = [n for n in base["filename"].drop_duplicates() if mapa.get(n)]
        if not nombres:
            st.warning("No hay imágenes disponibles con ese criterio.")
        else:
            rng = np.random.default_rng(int(semilla_g))
            elegidas = list(rng.choice(nombres, size=min(n_mostrar, len(nombres)), replace=False))
            cols = st.columns(3)
            for i, nombre in enumerate(elegidas):
                try:
                    img = ut.decodificar_imagen(imagenes[mapa[nombre]])
                except Exception:
                    continue
                dibujo = ut.dibujar_cajas(img, df[df["filename"] == nombre])
                cols[i % 3].image(dibujo, caption=ut.extraer_fuente(nombre)[:28] + "…", width=320)

# =============================================================================================
# 🎨 COLOR Y TEXTURA
# =============================================================================================
with tab_col:
    df_feat = obtener_features()
    if df_feat is None:
        st.info("Este análisis necesita las imágenes: cargalas en el Paso 1.")
    elif df_feat.empty:
        st.warning("No hay cajas con características para los filtros elegidos.")
    else:
        st.caption(f"Características calculadas para {len(df_feat)} hojas. Se miden sobre el recorte de cada caja "
                   "(estimaciones aproximadas: la caja incluye algo de fondo; ajustá el recorte y el umbral en la barra lateral).")
        feats_ok = [f for f in ut.FEATURES_COLOR if f in df_feat.columns]
        a, b = st.columns([1, 2])
        variable = a.selectbox("Variable", feats_ok, format_func=lambda v: ETQ.get(v, v))
        with b:
            fig = px.box(df_feat, x="clase", y=variable, color="clase", points="outliers", color_discrete_map=COLORES,
                         category_orders=ORDEN, title=f"{ETQ.get(variable, variable)} según la clase")
            fig.update_layout(showlegend=False, xaxis_title="Clase", yaxis_title=ETQ.get(variable, variable))
            grafico(fig, "g_col_box", alto=400)

        a, b = st.columns(2)
        with a:
            perfil = df_feat.groupby("clase")[feats_ok].mean()
            z = (perfil - perfil.mean()) / perfil.std(ddof=0).replace(0, 1)
            fig = px.imshow(z.T.fillna(0), color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, aspect="auto",
                            title="Perfil promedio por clase (color = valor relativo)")
            fig.update_traces(text=perfil.T.round(1).to_numpy(), texttemplate="%{text}")
            fig.update_layout(xaxis_title="Clase", yaxis_title="Característica", coloraxis_colorbar_title="z")
            grafico(fig, "g_col_perfil", alto=460)
        with b:
            corr = df_feat[feats_ok].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                            title="Correlación entre características")
            grafico(fig, "g_col_corr", alto=460)

        a, b = st.columns(2)
        with a:
            ex = st.selectbox("Eje X", feats_ok, index=feats_ok.index("indice_amarillamiento"),
                              format_func=lambda v: ETQ.get(v, v))
            ey = st.selectbox("Eje Y", feats_ok, index=feats_ok.index("pct_oscuro"), format_func=lambda v: ETQ.get(v, v))
            fig = px.scatter(df_feat, x=ex, y=ey, color="clase", opacity=0.65, color_discrete_map=COLORES,
                             category_orders=ORDEN, hover_data=["filename"], title="Relación entre dos características")
            fig.update_layout(xaxis_title=ETQ.get(ex, ex), yaxis_title=ETQ.get(ey, ey))
            grafico(fig, "g_col_scatter", alto=440)
        with b:
            m = df_feat.dropna(subset=feats_ok)
            if len(m) > 3:
                pca = PCA(n_components=2, random_state=42)
                comp = pca.fit_transform(StandardScaler().fit_transform(m[feats_ok]))
                dpca = pd.DataFrame({"PC1": comp[:, 0], "PC2": comp[:, 1], "clase": m["clase"].to_numpy()})
                fig = px.scatter(dpca, x="PC1", y="PC2", color="clase", opacity=0.65, color_discrete_map=COLORES,
                                 category_orders=ORDEN, title="Proyección PCA de las características de color y textura")
                fig.update_layout(xaxis_title=f"Componente 1 ({100 * pca.explained_variance_ratio_[0]:.1f} %)",
                                  yaxis_title=f"Componente 2 ({100 * pca.explained_variance_ratio_[1]:.1f} %)")
                grafico(fig, "g_col_pca", alto=500)

        st.download_button("⬇️ Descargar características (CSV)", df_feat.to_csv(index=False).encode("utf-8"),
                           "caracteristicas_hojas.csv", "text/csv")

# =============================================================================================
# 🤖 ENTRENAMIENTO
# =============================================================================================
with tab_ent:
    df_feat = obtener_features()
    if df_feat is None:
        st.info("El entrenamiento necesita las imágenes: cargalas en el Paso 1.")
    else:
        d_ok = df_feat[df_feat["clase"].isin(["Sana", "No sana"])]
        pos = d_ok[d_ok["clase"] == "No sana"]
        st.markdown("**Tarea:** clasificar cada hoja como *Sana* o *No sana* a partir de características de color y textura "
                    "extraídas de la imagen. La clase *Sin etiqueta* se excluye.")
        st.info(f"Datos disponibles: **{len(d_ok)}** hojas ({len(pos)} no sanas) provenientes de "
                f"**{d_ok['fuente'].nunique()}** fotos originales ({pos['fuente'].nunique()} con hojas no sanas). "
                "Con tan pocos casos positivos las métricas serán **inestables**: interpretalas con cautela.")

        a, b, c3 = st.columns(3)
        modelos_sel = a.multiselect("Modelos", ut.NOMBRES_MODELOS, default=["Random Forest", "Gradient Boosting"])
        tam_test = b.slider("Proporción de prueba", 0.10, 0.40, 0.25, 0.05)
        k_cv = c3.slider("Folds de validación cruzada", 3, 8, 5)
        a, b = st.columns([3, 1])
        cols_default = [f for f in ut.FEATURES_COLOR if f in d_ok.columns]
        cols_sel = a.multiselect("Características de entrada", cols_default + ut.FEATURES_GEOMETRIA, default=cols_default,
                                 format_func=lambda v: ETQ.get(v, v),
                                 help="Agregar la geometría de la caja puede hacer que el modelo 'aprenda' el tamaño de la "
                                      "caja en vez del estado de la hoja.")
        semilla = b.number_input("Semilla", 0, 9999, 42)

        if st.button("🚀 Entrenar y evaluar", type="primary"):
            if not modelos_sel or not cols_sel:
                st.error("Elegí al menos un modelo y una característica.")
            else:
                try:
                    with st.spinner("Entrenando (división por foto original)…"):
                        st.session_state["resultado"] = ut.entrenar_evaluar(
                            d_ok, cols_sel, modelos_sel, tam_test=tam_test, semilla=int(semilla), k_cv=k_cv)
                except Exception as e:
                    st.error(f"No se pudo entrenar: {e}")

        res = st.session_state.get("resultado")
        if res:
            st.divider()
            st.subheader("Resultados")
            c = st.columns(4)
            c[0].metric("Hojas de entrenamiento", res["n_train"], f"{res['pos_train']} no sanas", delta_color="off")
            c[1].metric("Hojas de prueba", res["n_test"], f"{res['pos_test']} no sanas", delta_color="off")
            c[2].metric("Fotos train / test", f"{res['fotos_train']} / {res['fotos_test']}")
            c[3].metric("Fotos compartidas", res["fotos_compartidas"], "sin fuga" if res["fotos_compartidas"] == 0 else "¡revisar!",
                        delta_color="off")

            umbral = st.slider("Umbral de decisión (probabilidad mínima para marcar «No sana»)", 0.05, 0.95, 0.50, 0.05)
            filas, matrices = [], {}
            for nombre, m in res["modelos"].items():
                met, cm = ut.metricas_binarias(res["y_test"], m["proba"], umbral)
                filas.append({"Modelo": nombre, **met})
                matrices[nombre] = cm
            tabla = pd.DataFrame(filas).set_index("Modelo")
            st.dataframe(tabla.round(3))
            st.caption(f"Mejor modelo por PR-AUC en la prueba: **{res['mejor']}**. Las métricas «No sana» tratan a esa clase como positiva.")

            nombres = list(res["modelos"])
            cols = st.columns(len(nombres))
            for col, nombre in zip(cols, nombres):
                cm = matrices[nombre]
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Sana", "No sana"],
                                y=["Sana", "No sana"], title=f"Confusión · {nombre}")
                fig.update_layout(xaxis_title="Predicha", yaxis_title="Real", coloraxis_showscale=False)
                fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=55, b=10))
                col.plotly_chart(fig, key=f"g_ent_cm_{nombre}")

            a, b = st.columns(2)
            fig_roc, fig_pr = go.Figure(), go.Figure()
            for nombre, m in res["modelos"].items():
                fpr, tpr, _ = roc_curve(res["y_test"], m["proba"])
                prec, rec, _ = precision_recall_curve(res["y_test"], m["proba"])
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=nombre))
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=nombre))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                         line=dict(dash="dash", color="gray")))
            base_pr = res["pos_test"] / max(res["n_test"], 1)
            fig_pr.add_trace(go.Scatter(x=[0, 1], y=[base_pr, base_pr], mode="lines", name="Azar (prevalencia)",
                                        line=dict(dash="dash", color="gray")))
            fig_roc.update_layout(title="Curva ROC (conjunto de prueba)", xaxis_title="Tasa de falsos positivos",
                                  yaxis_title="Tasa de verdaderos positivos")
            fig_pr.update_layout(title="Curva Precisión–Recall (clase No sana)", xaxis_title="Recall",
                                 yaxis_title="Precisión")
            with a:
                grafico(fig_roc, "g_ent_roc")
            with b:
                grafico(fig_pr, "g_ent_pr")

            if res["cv"] is not None:
                largo = res["cv"].melt(id_vars=["Modelo", "Fold"], var_name="Métrica", value_name="Valor")
                fig = px.box(largo, x="Métrica", y="Valor", color="Modelo", points="all",
                             title=f"Validación cruzada agrupada por foto ({res['k_cv']} folds)")
                fig.update_layout(xaxis_title="Métrica", yaxis_title="Valor por fold")
                grafico(fig, "g_ent_cv")
                resumen_cv = res["cv"].drop(columns="Fold").groupby("Modelo").agg(["mean", "std"]).round(3)
                st.dataframe(resumen_cv)
            else:
                st.warning("No hay suficientes fotos con hojas no sanas para una validación cruzada agrupada.")

            if res["importancias"] is not None:
                imp = res["importancias"].copy()
                imp["variable"] = imp["variable"].map(lambda v: ETQ.get(v, v))
                fig = px.bar(imp, x="importancia", y="variable", orientation="h", error_x="desvio",
                             title=f"Importancia por permutación (caída de PR-AUC) · {res['mejor']}",
                             color_discrete_sequence=["#4C78A8"])
                fig.update_layout(xaxis_title="Caída de PR-AUC al permutar la variable", yaxis_title="Variable")
                grafico(fig, "g_ent_imp", alto=450)

# =============================================================================================
# ✅ CALIDAD DEL DATASET
# =============================================================================================
with tab_cal:
    st.subheader("Chequeos de calidad")
    tam = df.groupby(["width", "height"]).size()
    sin_anot = len(set(k for k in imagenes) - set(v for v in mapa.values() if v)) if hay_imagenes else None
    sin_imagen = sum(1 for v in mapa.values() if v is None) if imagenes else None
    n_dup = int(df.duplicated(["filename", "xmin", "ymin", "xmax", "ymax"]).sum())
    sin_etq = int((df["clase"] == "Sin etiqueta").sum())
    chicas = int((df["pct_area_imagen"] < 1).sum())
    n_nombres = df.groupby("clase")["clase_original"].nunique()
    versiones_max = int(df.groupby("fuente")["filename"].nunique().max())

    chequeos = [
        ("Nombres de clase duplicados", "⚠️" if (n_nombres > 1).any() else "✅",
         "Hay clases con más de un nombre en el CSV (se unifican automáticamente)." if (n_nombres > 1).any()
         else "Cada clase tiene un único nombre."),
        ("Cajas «Sin etiqueta»", "⚠️" if sin_etq else "✅", f"{sin_etq} cajas sin clase asignada (se excluyen del entrenamiento)."),
        ("Cajas muy pequeñas (< 1 % de la imagen)", "⚠️" if chicas else "✅", f"{chicas} cajas; suelen ser errores de anotación."),
        ("Cajas duplicadas", "⚠️" if n_dup else "✅", f"{n_dup} cajas idénticas dentro de la misma imagen."),
        ("Fuga por aumentación", "⚠️" if versiones_max > 1 else "✅",
         f"Una foto original llega a tener {versiones_max} versiones: dividir siempre por foto original."),
        ("Tamaño de las imágenes", "✅" if len(tam) == 1 else "⚠️",
         ", ".join(f"{int(w)}×{int(h)} px ({n} cajas)" for (w, h), n in tam.items())),
    ]
    if sin_imagen is not None:
        chequeos.append(("Anotaciones sin imagen cargada", "⚠️" if sin_imagen else "✅",
                         f"{sin_imagen} imágenes del CSV no están en lo que cargaste."))
    if sin_anot is not None:
        chequeos.append(("Imágenes sin anotación", "⚠️" if sin_anot else "✅",
                         f"{sin_anot} imágenes cargadas no figuran en el CSV (sin hojas anotadas)."))
    st.dataframe(pd.DataFrame(chequeos, columns=["Chequeo", "Estado", "Detalle"]))

    st.markdown("**Recomendaciones para el informe**")
    st.markdown("- Declarar que las etiquetas disponibles son **sana / no sana** (detección); la *calidad comercial* de 4 clases "
                "y el *índice de rechazo* requieren etiquetas adicionales de acopio.\n"
                "- Dividir train/test **por foto original**, nunca por imagen.\n"
                "- Reportar F1, PR-AUC y accuracy balanceada; la accuracy simple es engañosa con tan pocos casos *No sana*.\n"
                "- Revisar manualmente las cajas «Sin etiqueta» y las muy pequeñas antes de entrenar.")

    a, b = st.columns(2)
    a.download_button("⬇️ Anotaciones limpias (CSV)", df.to_csv(index=False).encode("utf-8"), "anotaciones_limpias.csv", "text/csv")
    b.download_button("⬇️ Cajas por foto original (CSV)",
                      df.groupby("fuente").agg(imagenes=("filename", "nunique"), cajas=("clase", "size"),
                                               no_sanas=("clase", lambda s: int((s == "No sana").sum()))).reset_index()
                      .to_csv(index=False).encode("utf-8"), "resumen_por_foto.csv", "text/csv")
