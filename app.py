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

# Page configuration for a wider, cleaner layout
st.set_page_config(
    page_title="LexiScan - Auditoría Inteligente de Contratos",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS for aesthetics (organization, colors, cards - NO font changes)
st.markdown("""
    <style>
        /* Main background & container styling */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Custom card wrapper */
        .custom-card {
            background-color: #161B22;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #30363D;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
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
