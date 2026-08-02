import json
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)

# Umbral de confianza experimental (Parte 16 del notebook): por debajo de esto,
# se recomienda consultar a un dermatólogo en vez de mostrar un diagnóstico especifico.
CONFIDENCE_THRESHOLD = 0.70

# Umbral de "porcentaje de piel visible" (heurística clásica de color, no el
# modelo): si una foto tiene muy pocos píxeles con tono de piel, probablemente
# no es una foto útil para este clasificador (es un objeto, una pantalla, etc).
# Es un umbral experimental — bájalo si te da demasiados falsos positivos con
# fotos de piel legítimas (lesiones muy pigmentadas, mala iluminación, etc).
PIEL_MIN_PORCENTAJE = 0.12

GOOGLE_DRIVE_FILE_ID = "1HEFyoaMg77AMSfihagvEDOkKOKeFFvwb"
MODEL_PATH = "skin_disease_model.keras"
CLASS_NAMES_PATH = "clases.json"

# Preprocesamiento real usado en el entrenamiento (confirmado en el notebook):
# EfficientNetV2S. A diferencia de EfficientNet v1 (que no reescala los pixeles),
# EfficientNetV2 SI los reescala a rango -1..1.
preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input


# ----------------------------------------------------------------------
# TRADUCCIÓN AL ESPAÑOL DE LAS 12 CLASES
# ----------------------------------------------------------------------
LABELS_ES = {
    "Acne": "Acné",
    "Actinic_Keratosis": "Queratosis actínica",
    "Candidiasis": "Candidiasis",
    "Cancer_Piel": "Cáncer de piel",
    "Dano_Solar": "Daño solar",
    "Eczema": "Eccema",
    "Infestaciones_Picaduras": "Infestaciones y picaduras",
    "Lunares_Moles": "Lunares (nevos)",
    "Psoriasis": "Psoriasis",
    "Queratosis_Seborreica": "Queratosis seborreica",
    "Tinea_Hongos": "Tiña / hongos",
    "Verrugas": "Verrugas",
}


def nombre_legible(clase_original: str) -> str:
    if clase_original in LABELS_ES:
        return LABELS_ES[clase_original]
    return clase_original.replace("_", " ").title()


# ----------------------------------------------------------------------
# TOKENS DE DISEÑO
# ----------------------------------------------------------------------
# Paleta clínico-botánica: verde salvia + verde pino (confianza clínica)
# con una tira de calibración de tono de piel como elemento de marca —
# el mismo tipo de tarjeta de referencia de color que se usa en
# fotografía dermatológica clínica, aquí como firma visual del proyecto.
BG = "#EEF3F0"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F5F8F6"
INK = "#13322D"
INK_SOFT = "#4C6B63"
TEAL = "#0E6E63"
TEAL_DEEP = "#0B4A42"
AMBER = "#B5651D"
AMBER_BG = "#FBEEE3"

# Tira de calibración: espectro de tonos de piel (referencia clínica),
# reutilizada como firma de marca y como pista de los medidores de confianza.
ESPECTRO = ["#F5DEC6", "#E8C39C", "#D2A379", "#B47F55", "#8C5A3A", "#5A3826"]
ESPECTRO_CSS = ", ".join(ESPECTRO)

st.set_page_config(
    page_title="DermIA Honduras — Detección de enfermedades de la piel",
    page_icon="🔬",
    layout="wide",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html {{ color-scheme: light; }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"],
    .block-container {{
        background-color: {BG} !important;
        color: {INK} !important;
        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    }}
    [data-testid="stHeader"] {{ background-color: transparent !important; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}

    h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
    .stMarkdown, .stCaption, label, p, span {{ color: {INK}; }}

    /* --- Tira de calibración (firma de marca) --- */
    .calibracion {{
        display: flex;
        width: 100%;
        height: 10px;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 1.6rem;
        background: linear-gradient(90deg, {ESPECTRO_CSS});
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {TEAL};
        margin-bottom: 0.35rem;
    }}

    .marca {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        justify-content: center;
    }}
    .marca-titulo {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.75rem;
        color: {INK};
        letter-spacing: -0.01em;
    }}
    .sub-header {{
        font-size: 1.05rem;
        color: {INK_SOFT};
        text-align: center;
        max-width: 640px;
        margin: 0.4rem auto 2rem auto;
        line-height: 1.5;
    }}

    /* --- Bandeja de carga (evoca una charola de muestra clínica) --- */
    .upload-area {{
        border: 2px dashed {TEAL};
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        background: {SURFACE_ALT};
    }}

    /* --- Tarjeta de resultado --- */
    .result-card {{
        background: {SURFACE};
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        border: 1px solid #DCE7E2;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .disease-name {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {INK};
    }}
    .rank-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {INK_SOFT};
        letter-spacing: 0.06em;
    }}

    /* --- Medidor circular de confianza (conic-gradient) --- */
    .gauge {{
        position: relative;
        width: 54px;
        height: 54px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .gauge-inner {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: {SURFACE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        color: {INK};
    }}

    /* --- Aviso de triage / derivar a especialista --- */
    .triage {{
        background: {AMBER_BG};
        border-left: 4px solid {AMBER};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: #6B3B10;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }}

    .disclaimer {{
        background: {SURFACE_ALT};
        border-left: 4px solid {TEAL};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: {INK_SOFT};
        margin-top: 1.6rem;
        font-size: 0.88rem;
    }}

    /* --- Explicación visual (Grad-CAM) --- */
    .gradcam-caption {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {INK_SOFT};
        text-align: center;
        margin-top: 0.3rem;
    }}

    .streamlit-expanderHeader {{
        color: {INK} !important;
        background: {SURFACE_ALT} !important;
        border-radius: 10px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}

    /* Slider en tono teal */
    [data-testid="stSlider"] [role="slider"] {{ background-color: {TEAL} !important; }}
    [data-testid="stSlider"] > div > div > div > div {{ background-color: {TEAL} !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DESCARGA Y CARGA DEL MODELO (desde Google Drive)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# FILTRO DE ENTRADA: ¿esto parece piel? (heurística clásica de color,
# no el modelo — sirve para avisar si suben una foto de otra cosa)
# ----------------------------------------------------------------------
def porcentaje_pixeles_piel(imagen_pil) -> float:
    """Heurística clásica de detección de color de piel en espacio YCrCb.
    No reemplaza al modelo, solo sirve como filtro rápido de entrada:
    si casi nada de la imagen tiene tono de piel, probablemente subieron
    una foto de otra cosa (un objeto, una pantalla, un paisaje, etc)."""
    img = imagen_pil.convert("RGB").resize((150, 150))
    arr = np.array(img)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mascara = (y > 60) & (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)
    return float(mascara.mean())


# ----------------------------------------------------------------------
# TEST-TIME AUGMENTATION (misma técnica del notebook de entrenamiento)
# ----------------------------------------------------------------------
def predecir_con_tta(modelo, entrada_preprocesada):
    """Promedia la predicción de la imagen original y su espejo horizontal.
    El flip se aplica DESPUÉS de preprocess_input, lo cual es válido porque
    es puramente espacial y no interactúa con el escalado de píxeles."""
    probs_original = modelo.predict(entrada_preprocesada, verbose=0)
    entrada_flip = tf.image.flip_left_right(entrada_preprocesada)
    probs_flip = modelo.predict(entrada_flip, verbose=0)
    return (probs_original + probs_flip) / 2.0


# ----------------------------------------------------------------------
# GRAD-CAM (misma técnica del notebook de entrenamiento)
# ----------------------------------------------------------------------
def obtener_submodelo_base(modelo_completo):
    """Encuentra el sub-modelo EfficientNetV2S embebido dentro del modelo
    completo (fue agregado como una sola capa al construir el modelo)."""
    for layer in modelo_completo.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            return layer
    raise ValueError("No se encontró el sub-modelo base (EfficientNetV2S) dentro del modelo cargado.")


def encontrar_ultima_capa_conv(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(m.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No se encontró una capa convolucional en el modelo base.")


def calcular_gradcam(modelo_completo, base_model, ultima_capa_conv, entrada_preprocesada):
    """Recrea el forward pass del modelo completo en dos partes (base +
    capas de clasificación) para poder capturar los gradientes de la
    última capa convolucional respecto a la clase predicha."""
    grad_model = tf.keras.models.Model(
        [base_model.input], [base_model.get_layer(ultima_capa_conv).output, base_model.output]
    )
    capas_clasificacion = modelo_completo.layers[-5:]  # GAP, Dropout, Dense, Dropout, Dense(softmax)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(entrada_preprocesada)
        x = base_output
        for capa in capas_clasificacion:
            x = capa(x)
        pred_idx = tf.argmax(x[0])
        canal_clase = x[:, pred_idx]

    grads = tape.gradient(canal_clase, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superponer_heatmap(imagen_pil, heatmap, img_size):
    arr = np.array(imagen_pil.convert("RGB").resize(img_size)).astype("uint8")
    heatmap_resized = cv2.resize(heatmap, img_size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superpuesta = cv2.addWeighted(arr, 0.55, heatmap_color, 0.45, 0)
    return Image.fromarray(superpuesta)



    if Path(MODEL_PATH).exists():
        return
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource(show_spinner="Descargando y cargando el modelo (puede tardar la primera vez)...")
def cargar_modelo():
    descargar_modelo_si_hace_falta()
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        clases = json.load(f)

    # Grad-CAM es opcional: si la arquitectura cargada no coincide con lo
    # esperado (base EfficientNetV2S + 5 capas de clasificación), la app
    # sigue funcionando normal, solo sin la explicación visual.
    try:
        base_model = obtener_submodelo_base(modelo)
        ultima_capa_conv = encontrar_ultima_capa_conv(base_model)
    except Exception:
        base_model, ultima_capa_conv = None, None

    return modelo, clases, base_model, ultima_capa_conv


def preprocesar_imagen(imagen_pil):
    img = imagen_pil.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE)  # misma interpolacion (bilineal) que en el entrenamiento
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def render_gauge(prob: float) -> str:
    """Medidor circular hecho con conic-gradient, usando la misma paleta
    de la tira de calibración como pista de fondo."""
    grados = max(4, min(360, round(prob * 360)))
    color_relleno = TEAL if prob >= CONFIDENCE_THRESHOLD else AMBER
    return (
        f'<div class="gauge" style="background: conic-gradient(from -90deg, '
        f'{color_relleno} {grados}deg, #E2E9E5 {grados}deg 360deg);">'
        f'<div class="gauge-inner">{prob * 100:.0f}%</div></div>'
    )


def main():
    st.markdown(f'<div class="calibracion"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="marca">'
        '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="15" cy="15" r="10" stroke="#0E6E63" stroke-width="2.5"/>'
        '<line x1="22.5" y1="22.5" x2="30" y2="30" stroke="#0E6E63" stroke-width="2.5" stroke-linecap="round"/>'
        '</svg>'
        '<span class="marca-titulo">DermIA Honduras</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Análisis preliminar de afecciones de piel comunes en el contexto '
        'hondureño (clima tropical, alta radiación UV) mediante inteligencia artificial. '
        'Una herramienta de apoyo, no un reemplazo del criterio médico.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️  Cómo usar esta herramienta", expanded=False):
        st.markdown("""
        1. **Sube una imagen** nítida de la lesión cutánea (formatos JPG, JPEG o PNG).
        2. Ajusta cuántos diagnósticos diferenciales quieres ver.
        3. El modelo (EfficientNetV2S) analizará la imagen y mostrará las **enfermedades más probables** con su nivel de confianza.
        4. Si la imagen no parece tener piel visible, la herramienta te avisa antes de mostrar resultados.
        5. Si la confianza es baja, la herramienta va a recomendar consultar directamente con un dermatólogo.
        6. Debajo del resultado se muestra un **mapa de calor (Grad-CAM)** que resalta en qué zona de la imagen se fijó el modelo para decidir.
        7. Este sistema es **solo con fines educativos** y **no constituye un diagnóstico médico**.
        """)

    try:
        modelo, class_names, base_model, ultima_capa_conv = cargar_modelo()
    except Exception as e:
        st.error(
            "No se pudo descargar o cargar el modelo desde Google Drive. "
            "Revisa que el archivo siga compartido como 'Cualquier persona con el enlace' "
            f"y que '{CLASS_NAMES_PATH}' exista junto a app.py.\n\nError: {e}"
        )
        st.stop()

    col_control, _ = st.columns([1, 3])
    with col_control:
        top_k = st.slider(
            "Diagnósticos diferenciales a mostrar",
            min_value=1,
            max_value=min(10, len(class_names)),
            value=3,
        )

    st.markdown('<div class="eyebrow">Muestra</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    archivo = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo is not None:
        imagen = Image.open(archivo)

        col1, col2 = st.columns([1, 1.3], gap="large")

        with col1:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)

        pct_piel = porcentaje_pixeles_piel(imagen)
        if pct_piel < PIEL_MIN_PORCENTAJE:
            st.markdown(
                f'<div class="triage">🔍 <strong>Esta imagen tiene muy poco tono de piel visible '
                f'({pct_piel:.0%}).</strong> Este clasificador está entrenado solo para fotos de '
                f'lesiones cutáneas — si subiste otra cosa (un objeto, una pantalla, etc.), el '
                f'resultado no va a ser confiable. Verifica la foto antes de continuar.</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Analizando imagen..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = predecir_con_tta(modelo, entrada)[0]

        top_idx = np.argsort(predicciones)[-top_k:][::-1]
        confianza_principal = float(predicciones[top_idx[0]])

        with col2:
            st.markdown('<div class="eyebrow">Resultado</div>', unsafe_allow_html=True)

            if confianza_principal < CONFIDENCE_THRESHOLD:
                st.markdown(
                    f'<div class="triage">⚠️ <strong>Confianza baja '
                    f'({confianza_principal * 100:.0f}%).</strong> Por debajo del umbral mínimo '
                    f'({CONFIDENCE_THRESHOLD * 100:.0f}%) para un diagnóstico confiable. '
                    f'Se recomienda <strong>consultar directamente con un dermatólogo</strong>.</div>',
                    unsafe_allow_html=True,
                )

            for i, idx in enumerate(top_idx):
                clase = class_names[idx]
                prob = float(predicciones[idx])

                st.markdown(
                    f'<div class="result-card">{render_gauge(prob)}'
                    f'<div><div class="rank-label">DIFERENCIAL {i + 1:02d}</div>'
                    f'<div class="disease-name">{nombre_legible(clase)}</div></div></div>',
                    unsafe_allow_html=True,
                )

        if base_model is not None:
            st.markdown('<div class="eyebrow" style="margin-top:1.8rem;">Explicación visual (Grad-CAM)</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Generando mapa de calor..."):
                    heatmap = calcular_gradcam(modelo, base_model, ultima_capa_conv, entrada)
                    superpuesta = superponer_heatmap(imagen, heatmap, IMG_SIZE)

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.image(imagen.convert("RGB").resize(IMG_SIZE), use_container_width=True)
                    st.markdown('<div class="gradcam-caption">IMAGEN ORIGINAL</div>', unsafe_allow_html=True)
                with col_g2:
                    st.image(superpuesta, use_container_width=True)
                    st.markdown('<div class="gradcam-caption">ZONAS QUE MÁS INFLUYERON EN LA PREDICCIÓN</div>', unsafe_allow_html=True)
            except Exception:
                st.caption("No se pudo generar la explicación visual para esta imagen.")

        st.markdown(
            '<div class="disclaimer">Este resultado es generado por un modelo de inteligencia '
            'artificial con fines educativos y demostrativos. '
            '<strong>No constituye un diagnóstico médico.</strong> '
            'Ante cualquier duda, consulta a un dermatólogo.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Sube una imagen para comenzar el análisis.")

    st.markdown(f'<div class="calibracion" style="margin-top:2.4rem;"></div>', unsafe_allow_html=True)
    st.caption("DermIA Honduras · EfficientNetV2S · Modelo alojado en Google Drive")


if __name__ == "__main__":
    main()
