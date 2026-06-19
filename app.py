import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: white;
        border-radius: 10px;
        border: 1px solid #4a4a8a;
    }
    .stButton > button {
        background-color: #4a4a8a;
        color: white;
        border-radius: 10px;
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #6a6aaa;
    }
    .answer-box {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4a4a8a;
        color: white;
        margin-top: 20px;
    }
    .title-box {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1e2130, #2a2a5a);
        border-radius: 15px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-box">
    <h1>🤖 RAG Chatbot</h1>
    <p style="color: #aaaacc;">Upload a document and ask anything about it!</p>
</div>
""", unsafe_allow_html=True)

# HuggingFace token from environment or input
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    HF_TOKEN = st.text_input("🔑 Enter HuggingFace API Token", type="password", placeholder="hf_...")
    
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

    st.markdown("### 📄 Upload Your Document")
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file:
        with open("temp.txt", "w") as f:
            f.write(uploaded_file.read().decode("utf-8"))

        with st.spinner("📚 Processing your document..."):
            loader = TextLoader("temp.txt")
            documents = loader.load()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                task="text2text-generation",
                model_kwargs={"temperature": 0.5, "max_length": 256}
            )
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Document processed successfully!")
        st.markdown("### 💬 Ask a Question")
        question = st.text_input("Type your question here...", placeholder="What is this document about?")

        if st.button("🔍 Get Answer"):
            if question:
                with st.spinner("🤔 Thinking..."):
                    answer = qa_chain.run(question)
                st.markdown(f"""
                <div class="answer-box">
                    <b>🤖 Answer:</b><br><br>{answer}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please type a question first.")
else:
    st.info("👆 Please enter your HuggingFace API Token to get started.")