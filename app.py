import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stButton > button {
    background-color: #4a4a8a;
    color: white;
    border-radius: 10px;
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border: none;
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

st.markdown("""
<div class="title-box">
    <h1>🤖 RAG Chatbot</h1>
    <p style="color: #aaaacc;">Upload a document and ask anything about it!</p>
</div>
""", unsafe_allow_html=True)

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
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ Document processed successfully!")
    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Type your question here...", placeholder="What is this document about?")

    if st.button("🔍 Get Answer"):
        if question:
            with st.spinner("🤔 Searching document..."):
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

            st.markdown(f"""
            <div class="answer-box">
                <b>🤖 Most Relevant Answer from Document:</b><br><br>{context}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please type a question first.")