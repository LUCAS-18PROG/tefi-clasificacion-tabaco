import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42

def generar_dataset_tabaco(output_path="data/hojas_tabaco_features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(RANDOM_STATE)
    N = 647  # Más 3 duplicados intencionales = 650 registros crudos

    variedades = np.random.choice(['Virginia', 'Burley', 'Criollo'], size=N, p=[0.55, 0.30, 0.15])
    zonas = np.random.choice(['Perico', 'El Carmen', 'Monterrico'], size=N, p=[0.45, 0.35, 0.20])
    fuentes = np.random.choice(['Camara_Movil', 'Estudio_Controlado', 'Sintetica_Blender3D'], size=N, p=[0.50, 0.35, 0.15])
    perfil = np.random.choice([0, 1, 2, 3], size=N, p=[0.28, 0.34, 0.22, 0.16])

    largo_cm = np.where(perfil == 0, np.random.normal(52, 4, N),
               np.where(perfil == 1, np.random.normal(46, 5, N),
               np.where(perfil == 2, np.random.normal(39, 5, N), np.random.normal(32, 6, N))))
    largo_cm = np.clip(largo_cm, 18, 68)
    ancho_cm = largo_cm * np.random.uniform(0.42, 0.56, N)
    area_foliar_cm2 = np.round(largo_cm * ancho_cm * np.random.uniform(0.68, 0.78, N), 2)
    perimetro_cm = np.round(2.15 * np.sqrt(largo_cm**2 + ancho_cm**2) * np.random.uniform(0.95, 1.08, N), 2)

    tono_amarillo_pct = np.clip(np.where(perfil == 0, np.random.normal(72, 5, N),
                        np.where(perfil == 1, np.random.normal(58, 7, N),
                        np.where(perfil == 2, np.random.normal(40, 8, N), np.random.normal(24, 9, N)))), 5, 90)
    tono_verde_pct = np.clip(np.where(perfil == 0, np.random.normal(10, 4, N),
                     np.where(perfil == 1, np.random.normal(22, 6, N),
                     np.where(perfil == 2, np.random.normal(38, 9, N), np.random.normal(42, 14, N)))), 2, 85)

    total_av = tono_amarillo_pct + tono_verde_pct
    scale_av = np.where(total_av > 95, 95.0 / total_av, 1.0)
    tono_amarillo_pct = np.round(tono_amarillo_pct * scale_av, 2)
    tono_verde_pct = np.round(tono_verde_pct * scale_av, 2)
    tono_marron_pct = np.round(100.0 - tono_amarillo_pct - tono_verde_pct, 2)

    rugosidad_textura = np.round(np.clip(np.where(perfil == 0, np.random.normal(0.22, 0.05, N),
                        np.where(perfil == 1, np.random.normal(0.35, 0.07, N),
                        np.where(perfil == 2, np.random.normal(0.54, 0.09, N), np.random.normal(0.75, 0.11, N)))), 0.05, 0.99), 3)
    pct_grietas = np.round(np.clip(np.where(perfil == 0, np.random.exponential(0.5, N),
                  np.where(perfil == 1, np.random.exponential(1.4, N),
                  np.where(perfil == 2, np.random.normal(4.2, 1.5, N), np.random.normal(9.5, 2.8, N)))), 0.0, 25.0), 2)
    pct_perforaciones = np.round(np.clip(np.where(perfil == 0, np.random.exponential(0.3, N),
                        np.where(perfil == 1, np.random.exponential(1.0, N),
                        np.where(perfil == 2, np.random.normal(3.1, 1.2, N), np.random.normal(7.2, 2.4, N)))), 0.0, 20.0), 2)
    pct_moho = np.round(np.clip(np.where(perfil == 0, np.random.exponential(0.15, N),
               np.where(perfil == 1, np.random.exponential(0.45, N),
               np.where(perfil == 2, np.random.normal(1.8, 0.9, N), np.random.normal(5.6, 1.9, N)))), 0.0, 18.0), 2)

    indice_calidad = np.round(np.clip(
        52.0 + 0.42 * tono_amarillo_pct - 0.28 * tono_verde_pct + 0.012 * area_foliar_cm2
        - 12.0 * rugosidad_textura - 1.45 * pct_grietas - 1.65 * pct_perforaciones - 2.85 * pct_moho
        + np.random.normal(0, 2.5, N), 5.0, 99.5), 2)

    clase_calidad = []
    for idx_q, moho_v in zip(indice_calidad, pct_moho):
        if moho_v >= 3.8 or idx_q < 42:
            clase_calidad.append('Clase_4_Rechazo')
        elif idx_q < 60:
            clase_calidad.append('Clase_3_Tercera')
        elif idx_q < 76:
            clase_calidad.append('Clase_2_Segunda')
        else:
            clase_calidad.append('Clase_1_Primera')

    clase_calidad = np.array(clase_calidad)
    alerta_rechazo = np.where(np.isin(clase_calidad, ['Clase_3_Tercera', 'Clase_4_Rechazo']), 1, 0)

    df_raw = pd.DataFrame({
        'area_foliar_cm2': area_foliar_cm2,
        'perimetro_cm': perimetro_cm,
        'largo_cm': np.round(largo_cm, 2),
        'ancho_cm': np.round(ancho_cm, 2),
        'tono_verde_pct': tono_verde_pct,
        'tono_amarillo_pct': tono_amarillo_pct,
        'tono_marron_pct': tono_marron_pct,
        'rugosidad_textura': rugosidad_textura,
        'pct_grietas': pct_grietas,
        'pct_perforaciones': pct_perforaciones,
        'pct_moho': pct_moho,
        'variedad_tabaco': variedades,
        'zona_productor': zonas,
        'fuente_imagen': fuentes,
        'clase_calidad': clase_calidad,
        'alerta_rechazo': alerta_rechazo,
        'indice_calidad_comercial': indice_calidad
    })

    df_raw.loc[5, 'zona_productor'] = 'perico '
    df_raw.loc[18, 'zona_productor'] = 'EL CARMEN'
    df_raw.loc[42, 'variedad_tabaco'] = 'virginia'
    dups = df_raw.iloc[[10, 25, 100]].copy()
    df_raw = pd.concat([df_raw, dups], ignore_index=True)
    df_raw.to_csv(output_path, index=False)
    print(f"Dataset generado correctamente en {output_path}: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas.")

if __name__ == "__main__":
    generar_dataset_tabaco()
