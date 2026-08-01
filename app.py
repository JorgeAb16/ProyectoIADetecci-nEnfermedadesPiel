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
BG = "#EEF3F0"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F5F8F6"
BORDER = "#DCE7E2"
INK = "#13322D"
INK_SOFT = "#4C6B63"
INK_FAINT = "#7C948C"
TEAL = "#0E6E63"
TEAL_DEEP = "#0B4A42"
TEAL_TINT = "#E4F0EC"
AMBER = "#B5651D"
AMBER_BG = "#FBEEE3"

ESPECTRO = ["#F5DEC6", "#E8C39C", "#D2A379", "#B47F55", "#8C5A3A", "#5A3826"]
ESPECTRO_CSS = ", ".join(ESPECTRO)

st.set_page_config(
    page_title="DermIA Honduras — Detección de enfermedades de la piel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

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
    .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1120px; }}

    h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
    .stMarkdown, .stCaption, label, p, span {{ color: {INK}; }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}

    a:focus-visible, button:focus-visible, [role="button"]:focus-visible,
    input:focus-visible, textarea:focus-visible {{
        outline: 2px solid {TEAL} !important;
        outline-offset: 2px !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {TEAL_DEEP} !important;
        border-right: none;
    }}
    [data-testid="stSidebar"] * {{ color: #EAF3F0 !important; }}
    [data-testid="stSidebar"] .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.2rem;
    }}
    [data-testid="stSidebar"] .sidebar-brand-name {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        letter-spacing: -0.01em;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .sidebar-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9CC4BB !important;
        margin-bottom: 1.4rem;
    }}
    [data-testid="stSidebar"] .sidebar-section-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9CC4BB !important;
        margin: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.14);
        padding-top: 1.2rem;
    }}
    [data-testid="stSidebar"] .sidebar-step {{
        font-size: 0.86rem;
        line-height: 1.55;
        color: #D7E8E3 !important;
        margin-bottom: 0.55rem;
    }}
    [data-testid="stSidebar"] .sidebar-step b {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] .sidebar-legal {{
        font-size: 0.78rem;
        line-height: 1.55;
        color: #A9CAC3 !important;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 0.85rem 0.95rem;
        margin-top: 0.6rem;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{ background-color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {{ background-color: #6FBBAC !important; }}
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{ color: #A9CAC3 !important; }}

    .calibracion {{
        display: flex;
        width: 100%;
        height: 8px;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 2.1rem;
        background: linear-gradient(90deg, {ESPECTRO_CSS});
        box-shadow: 0 1px 0 rgba(19,50,45,0.05);
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {TEAL};
        margin-bottom: 0.5rem;
    }}

    .marca {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        justify-content: center;
    }}
    .marca-titulo {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.9rem;
        color: {INK};
        letter-spacing: -0.015em;
    }}
    .marca-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEAL_DEEP};
        background: {TEAL_TINT};
        border: 1px solid #C6E1D9;
        border-radius: 999px;
        padding: 0.22rem 0.65rem;
        margin-left: 0.35rem;
    }}
    .sub-header {{
        font-size: 1.06rem;
        color: {INK_SOFT};
        text-align: center;
        max-width: 620px;
        margin: 0.55rem auto 2.2rem auto;
        line-height: 1.55;
    }}

    .upload-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 1.6rem 1.7rem 1.3rem 1.7rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
        margin-bottom: 0.3rem;
    }}
    .upload-card-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.9rem;
    }}
    .upload-card-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: {INK};
    }}
    .upload-card-hint {{
        font-size: 0.82rem;
        color: {INK_FAINT};
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE_ALT} !important;
        border: 1.5px dashed #A9CDC3 !important;
        border-radius: 14px !important;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {TEAL} !important;
        background: {TEAL_TINT} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: {INK_SOFT} !important; }}
    [data-testid="baseButton-secondary"] {{
        border-radius: 8px !important;
        border-color: {TEAL} !important;
        color: {TEAL_DEEP} !important;
    }}

    .empty-state {{
        background: {SURFACE};
        border: 1px dashed {BORDER};
        border-radius: 20px;
        padding: 2.6rem 1.5rem;
        text-align: center;
        color: {INK_FAINT};
        margin-top: 0.5rem;
    }}
    .empty-state-icon {{ font-size: 1.8rem; margin-bottom: 0.6rem; }}
    .empty-state-title {{ font-weight: 600; color: {INK_SOFT}; font-size: 0.98rem; margin-bottom: 0.25rem; }}
    .empty-state-body {{ font-size: 0.86rem; max-width: 360px; margin: 0 auto; line-height: 1.5; }}

    .image-frame {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 0.7rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
    }}
    .image-frame img {{ border-radius: 13px !important; }}

    .result-card {{
        background: {SURFACE};
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        margin: 0.55rem 0;
        border: 1px solid {BORDER};
        display: flex;
        align-items: center;
        gap: 0.95rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.03);
        transition: border-color 0.15s ease, transform 0.15s ease;
        animation: aparecer 0.35s ease both;
    }}
    .result-card:hover {{
        border-color: #B7D6CC;
        transform: translateY(-1px);
    }}
    .result-card.principal {{
        border-color: {TEAL};
        background: linear-gradient(180deg, {TEAL_TINT} 0%, {SURFACE} 65%);
    }}
    @keyframes aparecer {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .rank-badge {{
        flex-shrink: 0;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        border: 1.5px solid {BORDER};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: {INK_FAINT};
        background: {SURFACE_ALT};
    }}
    .result-card.principal .rank-badge {{
        border-color: {TEAL};
        color: {TEAL_DEEP};
        background: #FFFFFF;
    }}

    .disease-name {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: {INK};
    }}
    .rank-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: {INK_FAINT};
        letter-spacing: 0.07em;
        margin-bottom: 0.05rem;
    }}

    .prob-track {{
        width: 100%;
        height: 4px;
        border-radius: 4px;
        background: #E2E9E5;
        margin-top: 0.4rem;
        overflow: hidden;
    }}
    .prob-fill {{ height: 100%; border-radius: 4px; }}

    .gauge {{
        position: relative;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .gauge-inner {{
        width: 39px;
        height: 39px;
        border-radius: 50%;
        background: {SURFACE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        font-weight: 600;
        color: {INK};
    }}

    .triage {{
        background: {AMBER_BG};
        border-left: 4px solid {AMBER};
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        color: #6B3B10;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }}
    .disclaimer {{
        background: {SURFACE_ALT};
        border-left: 4px solid {TEAL};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {INK_SOFT};
        margin-top: 1.7rem;
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    .chat-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 1.4rem 1.5rem 0.4rem 1.5rem;
        margin-top: 1.6rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
    }}
    .chat-card-head {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.2rem;
    }}
    .chat-card-title {{
        font-weight: 600;
        font-size: 1.0rem;
        color: {INK};
    }}
    .chat-card-sub {{
        font-size: 0.83rem;
        color: {INK_FAINT};
        margin-bottom: 0.9rem;
    }}
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 0.35rem 0 !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: {SURFACE_ALT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 0.65rem 0.95rem !important;
    }}
    [data-testid="stChatInput"] {{
        border-radius: 12px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: {SURFACE_ALT} !important;
        color: {INK} !important;
    }}

    .streamlit-expanderHeader {{
        color: {INK} !important;
        background: {SURFACE_ALT} !important;
        border-radius: 10px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}
    [data-testid="stSlider"] [role="slider"] {{ background-color: {TEAL} !important; }}
    [data-testid="stSlider"] > div > div > div > div {{ background-color: {TEAL} !important; }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; }}

    footer {{ visibility: hidden; }}
    .app-footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 2.4rem;
        padding-top: 1.1rem;
        border-top: 1px solid {BORDER};
        font-size: 0.8rem;
        color: {INK_FAINT};
    }}
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


def render_result_card(rank: int, clase: str, prob: float, es_principal: bool) -> str:
    color_relleno = TEAL if prob >= CONFIDENCE_THRESHOLD else AMBER
    clase_card = "result-card principal" if es_principal else "result-card"
    return (
        f'<div class="{clase_card}">'
        f'{render_gauge(prob)}'
        f'<div style="flex:1;">'
        f'<div class="rank-label">DIFERENCIAL {rank:02d}</div>'
        f'<div class="disease-name">{nombre_legible(clase)}</div>'
        f'<div class="prob-track"><div class="prob-fill" '
        f'style="width:{prob * 100:.0f}%; background:{color_relleno};"></div></div>'
        f'</div>'
        f'<div class="rank-badge">{rank:02d}</div>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# AGENTE DE PREGUNTAS (Hugging Face Inference Providers)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------
def render_sidebar(num_clases: int) -> int:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<svg width="26" height="26" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="15" cy="15" r="10" stroke="#FFFFFF" stroke-width="2.5"/>'
            '<line x1="22.5" y1="22.5" x2="30" y2="30" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
            '</svg>'
            '<span class="sidebar-brand-name">DermIA Honduras</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-tag">Análisis dermatológico con IA</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Cómo funciona</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-step"><b>1.</b> Sube una foto nítida de la lesión.</div>'
            '<div class="sidebar-step"><b>2.</b> El modelo (EfficientNetV2S) la analiza en segundos.</div>'
            '<div class="sidebar-step"><b>3.</b> Revisa los diagnósticos diferenciales más probables.</div>'
            '<div class="sidebar-step"><b>4.</b> Pregúntale al asistente lo que quieras saber sobre el resultado.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Preferencias</div>', unsafe_allow_html=True)
        top_k = st.slider(
            "Diagnósticos diferenciales a mostrar",
            min_value=1,
            max_value=min(10, num_clases),
            value=3,
        )

        st.markdown('<div class="sidebar-section-title">Aviso legal</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-legal">Esta herramienta tiene fines educativos y demostrativos. '
            'No constituye un diagnóstico médico ni sustituye la evaluación de un '
            'dermatólogo.</div>',
            unsafe_allow_html=True,
        )

        return top_k


def main():
    st.markdown('<div class="calibracion"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="marca">'
        '<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="15" cy="15" r="10" stroke="#0E6E63" stroke-width="2.5"/>'
        '<line x1="22.5" y1="22.5" x2="30" y2="30" stroke="#0E6E63" stroke-width="2.5" stroke-linecap="round"/>'
        '</svg>'
        '<span class="marca-titulo">DermIA Honduras</span>'
        '<span class="marca-badge">Beta</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Análisis preliminar de afecciones de piel comunes en el contexto '
        'hondureño (clima tropical, alta radiación UV) mediante inteligencia artificial. '
        'Una herramienta de apoyo, no un reemplazo del criterio médico.</div>',
        unsafe_allow_html=True,
    )

    try:
        modelo, class_names = cargar_modelo()
    except Exception as e:
        st.error(
            "No se pudo descargar o cargar el modelo desde Google Drive. "
            "Revisa que el archivo siga compartido como 'Cualquier persona con el enlace' "
            f"y que '{CLASS_NAMES_PATH}' exista junto a app.py.\n\nError: {e}"
        )
        st.stop()

    top_k = render_sidebar(len(class_names))

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-card-head">'
        '<span class="upload-card-title">Sube una imagen de la lesión</span>'
        '<span class="upload-card-hint">Formatos JPG, JPEG o PNG</span>'
        '</div>',
        unsafe_allow_html=True,
    )
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
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(imagen, caption="Imagen cargada", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
                    render_result_card(i + 1, clase, prob, es_principal=(i == 0)),
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="disclaimer">Este resultado es generado por un modelo de inteligencia '
            'artificial con fines educativos y demostrativos. '
            '<strong>No constituye un diagnóstico médico.</strong> '
            'Ante cualquier duda, consulta a un dermatólogo.</div>',
            unsafe_allow_html=True,
        )

        clase_principal = class_names[top_idx[0]]

        if st.session_state.get("clase_activa") != clase_principal:
            st.session_state["clase_activa"] = clase_principal
            st.session_state["mensajes_chat"] = [
                {
                    "role": "system",
                    "content": construir_instruccion_sistema(clase_principal, confianza_principal),
                }
            ]

        st.markdown('<div class="chat-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chat-card-head">'
            '<span class="chat-card-title">💬 Preguntas sobre este resultado</span>'
            '</div>'
            f'<div class="chat-card-sub">Pregúntale al asistente sobre '
            f'{nombre_legible(clase_principal).lower()}: causas, cuidados y cuándo consultar '
            'a un especialista.</div>',
            unsafe_allow_html=True,
        )

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
                avatar = "🧑" if mensaje["role"] == "user" else "🔬"
                with st.chat_message(mensaje["role"], avatar=avatar):
                    st.markdown(mensaje["content"])

            pregunta = st.chat_input(
                f"Pregunta algo sobre {nombre_legible(clase_principal).lower()}..."
            )

            if pregunta:
                st.session_state["mensajes_chat"].append({"role": "user", "content": pregunta})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(pregunta)

                with st.chat_message("assistant", avatar="🔬"):
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
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">🩺</div>'
            '<div class="empty-state-title">Aún no hay ninguna imagen cargada</div>'
            '<div class="empty-state-body">Sube una fotografía nítida de la piel para '
            'recibir un análisis preliminar y poder conversar con el asistente sobre '
            'el resultado.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="app-footer">'
        '<span>DermIA Honduras · EfficientNetV2S · Modelo alojado en Google Drive</span>'
        '<span>Herramienta educativa — no reemplaza consulta médica</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
