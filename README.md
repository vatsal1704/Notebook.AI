# 📑 Notebook.AI — Chat with Your PDFs Using Gemini + HuggingFace + FAISS

Notebook.AI is an intelligent, user-friendly, RAG-based (Retrieval Augmented Generation) PDF chatbot that lets you upload any PDF document and ask natural language questions about its content. Powered by **Gemini (Google's LLM)**, **HuggingFace Embeddings**, and **FAISS** (Facebook AI Similarity Search), it turns static documents into interactive conversations.

---

## 🚀 Live Demo

🔗 


---

## 🧠 How It Works

1. **Upload a PDF file** through the Streamlit interface.
2. The PDF is processed and text is chunked into manageable segments.
3. Each chunk is embedded using **HuggingFace's `all-MiniLM-L6-v2`** embedding model.
4. The chunks are stored and indexed in a **FAISS vector database** for efficient similarity-based retrieval.
5. When a question is asked, the most relevant chunks are retrieved and sent to **Gemini (via Google's GenerativeAI API)** for a context-aware response.

---

## 🛠️ Tech Stack

| Component       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Frontend**    | [Streamlit](https://streamlit.io/) - Simple and interactive UI              |
| **LLM**         | [Gemini (Google Generative AI)](https://deepmind.google/technologies/gemini/) for answering questions |
| **Embeddings**  | [HuggingFace](https://huggingface.co/) - `all-MiniLM-L6-v2` model for dense vector representation |
| **Vector Store**| [FAISS](https://github.com/facebookresearch/faiss) - High-speed similarity search engine |

---

## ✨ Features

- 📄 **Chat with any PDF** — Instantly load documents and ask questions.
- ⚡ **Fast and Scalable** — Efficient retrieval using FAISS vector index.
- 🤖 **Contextual Answers** — Gemini LLM gives accurate, context-aware answers.
- 🧩 **Modular Design** — Easy to swap models or databases as needed.
- 🧠 **RAG-Powered** — Combines retrieval with generative AI for deep comprehension.

---

## 🔧 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/notebook-ai.git
   cd notebook-ai
