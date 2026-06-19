import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.markdown("Ask anything from your uploaded document!")

hf_token = st.text_input("Enter your HuggingFace API Token", type="password")
uploaded_file = st.file_uploader("Upload a text file", type="txt")

if uploaded_file and hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    with open("temp.txt", "w") as f:
        f.write(uploaded_file.read().decode("utf-8"))

    loader = TextLoader("temp.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 256}
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.success(f"Answer: {answer}")