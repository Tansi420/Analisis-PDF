
import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Configuración de página
st.set_page_config(
    page_title="Archivo Cero | Laboratorio documental",
    page_icon="🔎",
    layout="wide"
)

# Estética visual
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101714 0%, #17241e 55%, #101714 100%);
        color: #e7eee9;
    }

    [data-testid="stSidebar"] {
        background-color: #17231d;
        border-right: 1px solid #30483a;
    }

    h1, h2, h3 {
        color: #d9f99d !important;
        letter-spacing: -0.5px;
    }

    p, label, .stMarkdown {
        color: #d0ddd4;
    }

    .hero {
        padding: 30px;
        background: linear-gradient(120deg, #20392c, #17251e);
        border: 1px solid #42664d;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .eyebrow {
        color: #a3e635;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        color: #f0fdf4;
        line-height: 1.15;
        margin: 12px 0;
    }

    .hero-description {
        font-size: 16px;
        color: #bdcfc2;
        max-width: 700px;
        line-height: 1.7;
    }

    .section-label {
        color: #a3e635;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .info-panel {
        background: #1b2b22;
        border: 1px solid #344b3b;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .info-title {
        color: #d9f99d;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-description {
        color: #b8c9bd;
        font-size: 14px;
        line-height: 1.6;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #101a14 !important;
        color: #f0fdf4 !important;
        border: 1px solid #496451 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] {
        background: #1b2b22;
        border: 1px dashed #66836c;
        border-radius: 14px;
        padding: 15px;
    }

    .stButton button {
        background: #a3e635;
        color: #17230e;
        border: none;
        border-radius: 9px;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: #1b2b22;
        padding: 15px;
        border: 1px solid #344b3b;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #d9f99d;
    }

    hr {
        border-color: #344b3b;
    }
</style>
""", unsafe_allow_html=True)


# Encabezado principal
st.markdown("""
<div class="hero">
    <div class="eyebrow">ARCHIVO CERO · UNIDAD DE ANÁLISIS</div>
    <div class="hero-title">Investiga más allá<br>de las páginas.</div>
    <div class="hero-description">
        Convierte documentos extensos en fuentes de respuestas.
        Carga un informe, explora su contenido y encuentra los datos
        que necesitas sin tener que leer cada página manualmente.
    </div>
</div>
""", unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    st.markdown("## 🔎 ARCHIVO CERO")
    st.caption("Laboratorio de investigación documental")
    st.divider()

    st.markdown("### ¿Qué puedes investigar?")
    st.write(
        "Consulta informes, revisa documentos académicos "
        "y localiza información relevante dentro de archivos PDF."
    )

    st.markdown("### Protocolo de uso")
    st.markdown("""
    **01** · Introduce tu clave de OpenAI.

    **02** · Carga el documento PDF.

    **03** · Escribe una pregunta específica.

    **04** · Examina la respuesta generada.
    """)

    st.divider()
    st.caption(f"Entorno Python · {platform.python_version()}")


# Área principal
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<div class="section-label">01 / ACCESO</div>',
                unsafe_allow_html=True)
    st.markdown("### Conecta el laboratorio")
    st.write(
        "Introduce tu clave de OpenAI para habilitar "
        "el análisis del documento."
    )

    ke = st.text_input(
        "Clave de API de OpenAI",
        type="password",
        placeholder="sk-..."
    )

    if ke:
        os.environ['OPENAI_API_KEY'] = ke
    else:
        st.warning("Introduce tu clave de API para continuar.")

    st.divider()

    st.markdown('<div class="section-label">02 / DOCUMENTO</div>',
                unsafe_allow_html=True)
    st.markdown("### Incorpora una fuente")
    st.write(
        "Selecciona un informe, artículo académico o documento "
        "que quieras explorar."
    )

    pdf = st.file_uploader(
        "Seleccionar documento PDF",
        type="pdf"
    )

    # Imagen original conservada
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=250)
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen: {e}")


with right:
    st.markdown('<div class="section-label">03 / EXPLORACIÓN</div>',
                unsafe_allow_html=True)
    st.markdown("### Sala de investigación")

    st.markdown("""
    <div class="info-panel">
        <div class="info-title">Tu documento, bajo la lupa.</div>
        <div class="info-description">
            El sistema extrae el texto, lo organiza en fragmentos
            y busca los pasajes relacionados con tu pregunta
            para construir una respuesta contextualizada.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Procesamiento del PDF
    if pdf is not None and ke:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            st.success("Documento incorporado al laboratorio.")

            metric1, metric2 = st.columns(2)
            metric1.metric("Caracteres extraídos", f"{len(text):,}")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )

            chunks = text_splitter.split_text(text)

            metric2.metric("Fragmentos creados", f"{len(chunks):,}")

            # Base de conocimiento
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.divider()

            st.markdown(
                '<div class="section-label">04 / CONSULTA</div>',
                unsafe_allow_html=True
            )
            st.markdown("### ¿Qué necesitas descubrir?")

            user_question = st.text_area(
                "Formula tu pregunta",
                placeholder=(
                    "Ej.: ¿Cuáles son las conclusiones principales "
                    "del informe?"
                ),
                height=120
            )

            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

                st.divider()
                st.markdown(
                    '<div class="section-label">RESULTADO DEL ANÁLISIS</div>',
                    unsafe_allow_html=True
                )
                st.markdown("### Hallazgos")
                st.markdown(response)

        except Exception as e:
            st.error(f"Error al procesar el PDF: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    elif pdf is not None and not ke:
        st.warning("Introduce tu clave de API de OpenAI para continuar.")

    else:
        st.info(
            "El laboratorio está listo. Incorpora un documento PDF "
            "y conecta tu clave de OpenAI para comenzar la investigación."
        )

st.divider()

st.caption(
    "ARCHIVO CERO · Laboratorio de investigación documental · "
    "Explora, pregunta y descubre."
)


