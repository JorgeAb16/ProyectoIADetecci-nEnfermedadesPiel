import json
import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

# Carga HF_TOKEN (y otras variables) desde un archivo .env en desarrollo
# local. En Streamlit Community Cloud no existe este archivo, por lo que
# el token se toma de st.secrets (ver obtener_hf_token más abajo).
load_dotenv()

# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)

# Umbral de confianza experimental (Parte 16 del notebook): por debajo de esto,
# se recomienda consultar a un dermatólogo en vez de mostrar un diagnóstico especifico.
CONFIDENCE_THRESHOLD = 0.70

GOOGLE_DRIVE_FILE_ID = "1HEFyoaMg77AMSfihagvEDOkKOKeFFvwb"
MODEL_PATH = "skin_disease_model.keras"
CLASS_NAMES_PATH = "clases.json"

# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL AGENTE DE PREGUNTAS (Hugging Face Inference Providers)
# ----------------------------------------------------------------------
# El modelo de chat es configurable por variable de entorno / secret para
# poder cambiarlo sin tocar código. Se usa un modelo instruct de propósito
# general disponible en el enrutador de Hugging Face.
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def obtener_hf_token() -> str:
    """
    Busca el token de Hugging Face primero en variables de entorno (.env,
    útil en local) y luego en st.secrets (usado en Streamlit Community Cloud).
    """
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return ""

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
def descargar_modelo_si_hace_falta():
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
    return modelo, clases


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


@st.cache_resource(show_spinner=False)
def obtener_cliente_hf():
    token = obtener_hf_token()
    if not token:
        return None
    return InferenceClient(api_key=token, provider="auto")


def construir_instruccion_sistema(clase_detectada: str, confianza: float) -> str:
    """
    Instrucción de sistema para el agente: se le informa la afección
    detectada por el modelo de visión, y se le pide comportarse como un
    asistente educativo, no como un médico.
    """
    return (
        "Eres un asistente educativo de salud de la piel, integrado en "
        "DermIA Honduras, una herramienta de análisis preliminar de "
        "imágenes dermatológicas orientada al contexto hondureño (clima "
        "tropical, alta radiación UV).\n\n"
        f"El modelo de visión por computadora analizó una imagen y detectó, "
        f"como diagnóstico diferencial más probable, "
        f"'{nombre_legible(clase_detectada)}' con un {confianza * 100:.0f}% "
        "de confianza.\n\n"
        "Tu función es responder en español las preguntas del usuario sobre "
        "esta afección: qué es, causas comunes, síntomas típicos, cuidados "
        "generales y cuándo se recomienda acudir a un dermatólogo. "
        "Responde de forma clara, breve y profesional.\n\n"
        "Reglas importantes:\n"
        "- NUNCA confirmes un diagnóstico definitivo: este es un resultado "
        "preliminar generado por IA, no un diagnóstico médico.\n"
        "- Recuerda al usuario, cuando sea pertinente, que debe consultar a "
        "un dermatólogo para una evaluación real.\n"
        "- No recomiendes medicamentos ni dosis específicas.\n"
        "- Si te preguntan algo fuera del tema de la piel o la afección "
        "detectada, puedes responder brevemente pero reorienta la "
        "conversación hacia el propósito de la herramienta."
    )


def responder_pregunta_agente(cliente: InferenceClient, historial: list, pregunta: str) -> str:
    """
    Envía la pregunta del usuario junto con el historial de la conversación
    al modelo de chat de Hugging Face y retorna la respuesta en texto.
    """
    mensajes = historial + [{"role": "user", "content": pregunta}]
    respuesta = cliente.chat.completions.create(
        model=HF_CHAT_MODEL,
        messages=mensajes,
        max_tokens=500,
        temperature=0.4,
    )
    return respuesta.choices[0].message.content


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
        4. Si la confianza es baja, la herramienta va a recomendar consultar directamente con un dermatólogo.
        5. Este sistema es **solo con fines educativos** y **no constituye un diagnóstico médico**.
        """)

    try:
        modelo, class_names = cargar_modelo()
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

        with st.spinner("Analizando imagen..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = modelo.predict(entrada, verbose=0)[0]

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

        st.markdown(
            '<div class="disclaimer">Este resultado es generado por un modelo de inteligencia '
            'artificial con fines educativos y demostrativos. '
            '<strong>No constituye un diagnóstico médico.</strong> '
            'Ante cualquier duda, consulta a un dermatólogo.</div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # AGENTE CONVERSACIONAL (Hugging Face) SOBRE LA AFECCIÓN DETECTADA
        # ------------------------------------------------------------
        clase_principal = class_names[top_idx[0]]
        clave_conversacion = f"chat_{clase_principal}"

        # Si cambia la afección detectada (nueva imagen), se reinicia el chat
        if st.session_state.get("clase_activa") != clase_principal:
            st.session_state["clase_activa"] = clase_principal
            st.session_state["mensajes_chat"] = [
                {
                    "role": "system",
                    "content": construir_instruccion_sistema(clase_principal, confianza_principal),
                }
            ]

        st.markdown('<div class="eyebrow">Preguntas sobre este resultado</div>', unsafe_allow_html=True)

        cliente_hf = obtener_cliente_hf()

        if cliente_hf is None:
            st.info(
                "Para habilitar el asistente de preguntas, configura la variable "
                "`HF_TOKEN` (token de Hugging Face) en tu archivo `.env` local "
                "o en los secrets de Streamlit."
            )
        else:
            for mensaje in st.session_state["mensajes_chat"]:
                if mensaje["role"] == "system":
                    continue
                with st.chat_message(mensaje["role"]):
                    st.markdown(mensaje["content"])

            pregunta = st.chat_input(
                f"Pregunta algo sobre {nombre_legible(clase_principal).lower()}..."
            )

            if pregunta:
                st.session_state["mensajes_chat"].append({"role": "user", "content": pregunta})
                with st.chat_message("user"):
                    st.markdown(pregunta)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            respuesta_texto = responder_pregunta_agente(
                                cliente_hf, st.session_state["mensajes_chat"][:-1], pregunta
                            )
                        except Exception as e:
                            respuesta_texto = (
                                "No fue posible obtener respuesta del asistente en este "
                                f"momento. Detalle técnico: {e}"
                            )
                    st.markdown(respuesta_texto)

                st.session_state["mensajes_chat"].append(
                    {"role": "assistant", "content": respuesta_texto}
                )
    else:
        st.info("Sube una imagen para comenzar el análisis.")

    st.markdown(f'<div class="calibracion" style="margin-top:2.4rem;"></div>', unsafe_allow_html=True)
    st.caption("DermIA Honduras · EfficientNetV2S · Modelo alojado en Google Drive")


if __name__ == "__main__":
    main()
