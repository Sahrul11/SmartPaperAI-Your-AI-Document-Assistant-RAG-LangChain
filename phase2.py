import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Set up a modern page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for a cool look
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #0099ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("📚✨ SmartPaperAI: Your AI Document Assistant 🚀")

# file uploader widget
pdf_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

prompt = st.chat_input("💬 Enter your prompt...")

# Setup a session state to store the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history with icons
for message in st.session_state.messages:
    role = message["role"]
    icon = "👤" if role == "user" else "🤖"
    st.markdown(f"**{icon} {role.capitalize()}:** {message['content']}")

@st.cache_resource
def get_vectorstore(pdf_bytes):
    # Write the uploaded PDF bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes.getbuffer())
        tmp_path = tmp.name

    # Load the PDF file using the temporary file path
    loader = [PyPDFLoader(tmp_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    ).from_loaders(loader)
    return index.vectorstore

if prompt:
    st.markdown(f"**👤 User:** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything, you always give the best,
           the most accurate and most precise answers. Answer the following Question: {user_prompt}.
           Start the answer directly. No small talk please"""
    )

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=1000,
    )

    try:
        if pdf_file is None:
            st.error("❗ Please upload a PDF file to continue.")
        else:
            with st.spinner("📝 Processing your PDF..."):
                vectorstore = get_vectorstore(pdf_file)
            if vectorstore is None:
                raise ValueError("Vectorstore is not available. Please check the PDF file.")

            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )
            with st.spinner("🤔 Fetching answer..."):
                result = chain({"query": prompt})
            response = result["result"]
            st.markdown(f"**🤖 Assistant:** {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")