import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------🔐 API CONFIGURATION ------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------🧠 EMBEDDING MODEL ------------------------
@st.cache_resource(show_spinner="🔄 Loading the HuggingFace Embedding Model...")
def embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------------🎯 STREAMLIT FRONT-END ------------------------
st.title("📑NOTEBOOK:blue[.AI]")
st.markdown("### 🤖  RAG-Based PDF Q&A System: Powered by HuggingFace Embeddings + FAISS + Gemini")
st.markdown("---")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF Document", type=["pdf"])

# ------------------------📄 PDF PROCESSING ------------------------
if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)
    for index, page in enumerate(pdf.pages):
        context = page.extract_text()
        if context:
            raw_text += context

    # Ensure text exists
    if raw_text.strip():
        document = Document(page_content=raw_text)

        # Chunking the document
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([document])

        # Prepare FAISS vector DB
        texts = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(texts, embedding_model())
        retriever = vector_db.as_retriever()

        # ✅ User Input Form
        st.success("✅ Document processed successfully. Ask your questions below!")

        with st.form(key="query_form", clear_on_submit=True):
            user_input = st.text_input("💬 Enter your question:", key="user_input")
            submitted = st.form_submit_button("🔍 Submit")

        if submitted and user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("🧠 Analysing and thinking..."):
                # Retrieve relevant chunks
                retrieved_docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # Prompt formatting
                prompt = f"""You are an expert assistant. Use the context below to answer the query.
If unsure, just say - "I Don't Know".
Context: {context}
User Query: {user_input}
Answer:"""

                # Generate the answer
                response = model.generate_content(prompt)

                # Show response
                st.markdown("### 📝 Answer:")
                st.write(response.text)

else:
    st.warning("⚠️ Please upload a PDF file to begin.")