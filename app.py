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

# Page configuration for a wide, immersive dashboard layout
st.set_page_config(
    page_title="LexiScan - Auditoría Inteligente de Contratos",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced aesthetics (modern cards, layout reorganization, rich dark theme - NO font changes)
st.markdown("""
    <style>
        /* Main background & overall layout */
        .stApp {
            background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
            color: #f3f4f6;
        }
        
        /* Sidebar custom polish */
        [data-testid="stSidebar"] {
            background-color: #070a10;
            border-right: 1px solid #1f293d;
        }
        
        /* Card container style */
        .lexi-card {
            background-color: rgba(26, 32, 44, 0.75);
            backdrop-filter: blur(10px);
            padding: 1.8rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 1.5rem;
        }
        
        /* Header styling banner */
        .hero-banner {
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            padding: 2rem;
            border-radius: 16px;
            border-left: 6px solid #6366f1;
            margin-bottom: 2rem;
        }

        /* Inputs & Textareas enhancement */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
        }
        
        /* File uploader container box */
        [data-testid="stFileUploadDropzone"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 2px dashed #475569 !important;
            border-radius: 14px !important;
        }
        
        /* Alerts & Info blocks */
        .stAlert {
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar information with enhanced layout
with st.sidebar:
    st.markdown("### 🛡️ Centro de Control LexiScan")
    st.markdown("Asistente forense avanzado para la detección de cláusulas críticas, riesgos legales ocultos y pasivos en contratos.")
    st.markdown("---")
    st.info("💡 **Consejo:** Asegúrate de que tu clave de OpenAI esté activa antes de cargar el documento contractual.")
    st.markdown("---")
    st.caption("🔒 Conexión cifrada de extremo a extremo.")

# App Header Banner (Hero layout)
st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0; font-size: 2.2rem; color: #ffffff;">⚖️ LexiScan: Auditoría y Análisis de Contratos Legales</h1>
        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.05rem;">Entorno de Inteligencia Artificial Aplicada a la Revisión Documental y Contractual</p>
    </div>
""", unsafe_allow_html=True)

# Layout division in columns for top section (API Key & Image)
col_main, col_side = st.columns([2, 1], gap="large")

with col_side:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_container_width=True)
    except Exception as e:
        pass

with col_main:
    # Get API key from user inside a styled block
    st.markdown("#### 1. Credenciales de Acceso")
    ke = st.text_input('Ingresa tu Clave de OpenAI', type="password", placeholder="sk-proj-...")
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("✅ Clave configurada correctamente.")
    else:
        st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar.")
    
    st.write("Versión de Python en entorno:", platform.python_version())

st.markdown("---")

# PDF uploader section wrapped in structure
st.markdown("#### 2. Carga del Expediente o Contrato")
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        with st.spinner("Procesando y digitalizando documento legal..."):
            # Extract text from PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
        st.info(f"📊 Texto extraído con éxito: {len(text)} caracteres registrados.")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🔍 Documento segmentado en {len(chunks)} fragmentos normativos.")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        st.markdown("---")
        
        # User question interface inside a dedicated card container
        st.markdown("""
            <div class="lexi-card">
                <h3 style="margin-top: 0; color: #e2e8f0;">💬 Interrogatorio de Cláusulas y Riesgos</h3>
                <p style="color: #94a3b8; font-size: 0.95rem;">Realiza consultas específicas sobre obligaciones, plazos de entrega, multas o jurisdicción.</p>
            </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_area(" ", placeholder="Ej: ¿Cuáles son las causales de terminación anticipada de este acuerdo?")
        
        # Process question when submitted
        if user_question:
            with st.spinner("Buscando respuestas en las cláusulas del documento..."):
                docs = knowledge_base.similarity_search(user_question)
                
                # Use a current model instead of deprecated text-davinci-003
                llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
                
                # Load QA chain
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Run the chain
                response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response inside a styled result container
            st.markdown("---")
            st.markdown("### 📋 Dictamen de la Cláusula:")
            st.markdown(f"""
                <div class="lexi-card" style="border-left: 4px solid #10b981; background-color: rgba(15, 23, 42, 0.9);">
                    {response}
                </div>
            """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📂 Por favor carga un archivo PDF para comenzar la auditoría.")        
        /* Sidebar styling override */
        [data-testid="stSidebar"] {
            background-color: #111418;
            border-right: 1px solid #30363D;
        }
        
        /* Input and file uploader accents */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: #0D1117;
            color: #FAFAFA;
            border: 1px solid #30363D;
            border-radius: 8px;
        }
        
        /* Metric/Info boxes */
        .stAlert {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and presentation
st.title('⚖️ LexiScan: Auditoría y Análisis de Contratos Legales')
st.write("Entorno de Procesamiento Legal | Versión de Motor:", platform.python_version())

# Load and display image inside a neat container
try:
    image = Image.open('Chat_pdf.png')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=280)
except Exception as e:
    pass

# Sidebar information
with st.sidebar:
    st.subheader("🛡️ Panel de Control Legal")
    st.markdown("Este asistente forense analiza cláusulas, riesgos ocultos e inconsistencias legales en el documento contractual cargado.")
    st.markdown("---")
    st.markdown("💡 *Sube el contrato, introduce tu credencial y haz consultas directas sobre obligaciones, plazos y multas.*")

# Get API key from user
ke = st.text_input('🔑 Clave de Acceso OpenAI (API Key)', type="password", placeholder="sk-...")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("⚠️ Por favor ingresa tu credencial de API para habilitar el motor de análisis legal.")

st.markdown("---")

# PDF uploader container
st.subheader("📁 Repositorio de Documentos Legales")
pdf = st.file_uploader("Carga el contrato o documento legal en formato PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📊 Texto analizado del documento: {len(text)} caracteres procesados.")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🔍 Documento segmentado en {len(chunks)} cláusulas y fragmentos clave.")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        st.markdown("---")
        
        # User question interface
        st.subheader("💬 Interrogatorio de Cláusulas y Riesgos")
        user_question = st.text_area(" ", placeholder="Ej: ¿Cuáles son las penalizaciones por rescisión anticipada de este contrato?")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### 📋 Dictamen de la Cláusula:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"❌ Error crítico en el análisis del contrato: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar con la auditoría.")
else:
    st.info("📂 Carga un archivo PDF de contrato o documento legal para iniciar la revisión.")
