# 🔎 RAG Research Paper Question Answering

> A Hybrid Retrieval-Augmented Generation system for answering questions from research papers using semantic retrieval, keyword search, reranking, and LLM-based generation.

## 🚀 Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline using research papers collected from arXiv.

The system retrieves relevant paper content using a **hybrid retrieval approach**, reranks the retrieved chunks using a **CrossEncoder**, and generates context-grounded answers using a **Groq-hosted LLM**.

The project also evaluates generated answers using **RAGAS**.

## 🧠 Architecture

```text
                 Research Papers
                       │
                       ▼
                PDF Text Extraction
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
              Sentence Transformers
                       │
                       ▼
              ┌───────────────────┐
              │   Hybrid Search   │
              │                   │
              │   FAISS + BM25    │
              └─────────┬─────────┘
                        │
                        ▼
                CrossEncoder
                  Reranking
                        │
                        ▼
                   Top Chunks
                        │
                        ▼
                   Groq LLM
                        │
                        ▼
                Grounded Answer
                        │
                        ▼
                 RAGAS Evaluation
```

## ✨ Features

* 📄 Automatic research paper collection from arXiv
* 🔤 PDF text extraction using PyMuPDF
* ✂️ Recursive text chunking
* 🧠 Sentence Transformer embeddings
* ⚡ FAISS vector similarity search
* 🔎 BM25 keyword-based retrieval
* 🔀 Hybrid retrieval combining semantic and keyword search
* 🎯 CrossEncoder-based reranking
* 🤖 Groq LLM-based answer generation
* 📊 RAGAS evaluation
* 🛡️ Context-grounded answering to reduce unsupported responses
* 🔍 Cosine similarity based vector retrieval
* 🔄 Comparison-question handling
* 📚 Small-document handling

## 🛠️ Tech Stack

| Technology            | Purpose               |
| --------------------- | --------------------- |
| Python                | Core implementation   |
| arXiv                 | Research paper source |
| PyMuPDF               | PDF text extraction   |
| Sentence Transformers | Text embeddings       |
| FAISS                 | Vector retrieval      |
| BM25                  | Keyword retrieval     |
| CrossEncoder          | Reranking             |
| Groq                  | LLM generation        |
| LangChain             | LLM integration       |
| RAGAS                 | RAG evaluation        |

## 📊 Current Dataset

The current pipeline retrieves **15 research papers** related to Retrieval-Augmented Generation.

After preprocessing:

* **Research papers:** 15
* **Total chunks:** 1,634
* **Embedding model:** `all-MiniLM-L6-v2`
* **Vector store:** FAISS
* **Keyword retriever:** BM25

## 📈 Evaluation

The current evaluation uses:

* **Faithfulness:** 0.9333
* **Answer Relevancy:** 0.8488

These results were obtained from the current evaluation run on a small test set of five questions.

## 💡 Example

### Question

> What is retrieval augmented generation and why is it used?

### System

The system retrieves relevant research-paper chunks, reranks them, and provides the selected context to the LLM.

### Answer

The generated answer explains RAG as a combination of a pretrained language model and an external retrieval system, allowing the model to use external and domain-specific information at inference time.

## 📁 Project Structure

```text
RAG-Research-Paper-QA/
│
├── RAG_Research_Paper_QA.ipynb
└── README.md
```

The project structure will be expanded as the system is further modularized.

## 🔮 Future Improvements

* [ ] Improve retrieval quality
* [ ] Add larger evaluation dataset
* [ ] Add retrieval-specific evaluation metrics
* [ ] Add better metadata handling
* [ ] Add citation-aware answers
* [ ] Add a web interface
* [ ] Deploy the application
* [ ] Optimize latency and token usage

## ⚠️ Notes

The project is currently under active development.

API keys should be stored securely using environment variables or notebook secrets and should **never be committed to GitHub**.

## 👩‍💻 Author

**Anusha Panicker**

B.Tech AI & ML Student
Aspiring AI Engineer
