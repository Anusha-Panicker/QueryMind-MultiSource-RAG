<h1 align="center">🧠 QueryMind</h1>

<p align="center">
  <strong>Multi-Source Retrieval-Augmented Generation for Research & Knowledge Discovery</strong>
</p>

<p align="center">
  Retrieve • Rank • Understand
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-00A67E?style=for-the-badge">
  <img src="https://img.shields.io/badge/BM25-Hybrid_Search-6C63FF?style=for-the-badge">
  <img src="https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge">
</p>

---

<p align="center">
  <em>
    An AI-powered RAG system for discovering, processing, retrieving, and
    understanding information from research papers, PDFs, and YouTube transcripts.
  </em>
</p>

---


## 🔎 What is QueryMind?

**QueryMind** is an end-to-end **Retrieval-Augmented Generation (RAG)** application built to help users search, understand, and question information from multiple sources.

Instead of sending an entire document directly to an LLM, QueryMind first **retrieves the most relevant information**, ranks it, and then provides that context to the language model.

The system combines:

* 🔎 Research paper discovery
* 📄 PDF document processing
* ▶️ YouTube transcript processing
* 🧠 Semantic vector retrieval
* 🔤 Keyword-based retrieval
* 🔀 Hybrid search
* 🎯 CrossEncoder reranking
* 🤖 LLM-based generation
* 📝 Content summarization

The goal is simple:

> **Find the right information first. Then let the LLM understand it.**

---

# ✨ Key Features

## 🔬 1. Research Paper Discovery

Search for a research topic directly from QueryMind.

A single topic search can retrieve **up to 10 relevant research papers**, download their content, process the PDFs, and add them to the retrieval pipeline.

**Workflow:**

```text
Topic
  ↓
Research Paper Search
  ↓
Retrieve up to 10 Papers
  ↓
Download PDFs
  ↓
Extract Text
  ↓
Chunk Documents
  ↓
Build Search Index
```

---

## 📄 2. PDF Intelligence

Upload your own PDF documents and interact with their content.

QueryMind:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS + BM25
 ↓
Reranking
 ↓
LLM
```

This allows users to ask questions about research papers, notes, reports, documentation, and other text-based PDF content.

---

## ▶️ 3. YouTube Transcript Processing

QueryMind can process YouTube videos through their available transcripts.

```text
YouTube URL
     ↓
Transcript Extraction
     ↓
Text Cleaning
     ↓
Chunking
     ↓
Indexing
     ↓
Question Answering / Summarization
```

This makes the same RAG pipeline usable for both written and spoken knowledge.

---

# 🧠 Advanced Retrieval Pipeline

The main strength of QueryMind is its **multi-stage retrieval architecture**.

A basic RAG system may rely on only vector similarity.

QueryMind combines multiple retrieval techniques:

```text
                         USER QUERY
                              │
                              ▼
                    ┌─────────────────┐
                    │ Query Processing │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌───────────────┐         ┌───────────────┐
        │ FAISS Search  │         │  BM25 Search  │
        │   Semantic    │         │    Keyword    │
        │   Retrieval   │         │   Retrieval   │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                   ┌──────────────────┐
                   │ Hybrid Retrieval │
                   └────────┬─────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ CrossEncoder       │
                  │    Reranking       │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Relevant Context   │
                  └─────────┬──────────┘
                            │
                            ▼
                     ┌────────────┐
                     │  Groq LLM  │
                     └─────┬──────┘
                           │
                  ┌────────┴─────────┐
                  ▼                  ▼
             Answer              Summary
```

---

# ⚡ Why Hybrid Retrieval?

QueryMind does not depend on a single search method.

### 🧠 Semantic Retrieval — FAISS

Sentence Transformer embeddings convert text into vectors.

FAISS then searches for chunks that are **semantically similar** to the user's question.

This helps when the query and document use different words but express similar meanings.

### 🔤 Keyword Retrieval — BM25

BM25 focuses on important words and exact lexical matches.

This is useful for:

* Technical terminology
* Model names
* Algorithms
* Dataset names
* Specific concepts
* Exact phrases

### 🔀 Hybrid Retrieval

QueryMind combines both approaches.

```text
Semantic Search
       +
Keyword Search
       ↓
Hybrid Candidate Set
       ↓
CrossEncoder Reranking
```

This gives the system access to both **meaning-based retrieval** and **keyword-based retrieval**.

---

# 🎯 CrossEncoder Reranking

Retrieval alone does not guarantee that the first results are the most useful.

QueryMind therefore uses a **CrossEncoder** to evaluate the retrieved candidates and reorder them according to query-document relevance.

```text
FAISS + BM25
     ↓
Candidate Chunks
     ↓
CrossEncoder
     ↓
Relevance Scoring
     ↓
Top Relevant Context
```

This creates a two-stage retrieval architecture:

**Stage 1 — Recall**

> Retrieve potentially relevant information.

**Stage 2 — Precision**

> Rerank the candidates and select the most relevant context.

---

# 🤖 Grounded Question Answering

Once relevant chunks are retrieved, QueryMind passes the selected context to the LLM.

The generation pipeline follows:

```text
User Question
      ↓
Retrieve Relevant Chunks
      ↓
Hybrid Search
      ↓
CrossEncoder Reranking
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Context-Grounded Answer
```

The system is designed to make the LLM answer using the retrieved context rather than treating the model as a standalone knowledge source.

If the required information is not available in the retrieved context, the application is designed to avoid confidently generating unsupported information.

---

# 📝 Summarization

QueryMind also provides summarization capabilities for loaded content.

Users can summarize:

* 📄 PDF documents
* ▶️ YouTube transcripts
* 🔬 Retrieved research-paper content

The same LLM infrastructure is used to transform long-form content into a concise representation.

---

# 📚 Multi-Source Knowledge

QueryMind currently supports three major input types:

| Source             | Processing Pipeline                           |
| ------------------ | --------------------------------------------- |
| 🔬 Research Papers | Topic → Papers → PDFs → Text → Chunks → Index |
| 📄 PDF Documents   | Upload → Extract → Chunk → Index              |
| ▶️ YouTube Videos  | URL → Transcript → Chunk → Index              |

The architecture allows information from different sources to be processed through the same retrieval pipeline.

---

# 🛠️ Technology Stack

| Technology                   | Purpose                     |
| ---------------------------- | --------------------------- |
| **Python**                   | Core implementation         |
| **Streamlit**                | Interactive web interface   |
| **PyMuPDF**                  | PDF text extraction         |
| **Sentence Transformers**    | Semantic embeddings         |
| **FAISS**                    | Vector similarity search    |
| **Rank-BM25**                | Keyword retrieval           |
| **CrossEncoder**             | Candidate reranking         |
| **Groq**                     | LLM inference               |
| **LangChain Text Splitters** | Recursive document chunking |
| **arXiv / Semantic Scholar** | Research paper discovery    |
| **YouTube Transcript API**   | Transcript extraction       |

---

# 📂 Project Structure

```text
QueryMind-MultiSource-RAG/
│
├── src/
│   ├── app.py
│   ├── engine.py
│   └── test_engine.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### `app.py`

Contains the Streamlit application and user-facing interface.

### `engine.py`

Contains the core RAG and retrieval logic including document processing, retrieval, reranking, and generation.

### `test_engine.py`

Contains testing utilities for the retrieval engine.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Anusha-Panicker/QueryMind-MultiSource-RAG.git

cd QueryMind-MultiSource-RAG
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

Keep your API keys private.

**Never commit `.env` or API credentials to GitHub.**

---

# ▶️ Run QueryMind

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Then open the local Streamlit URL shown in your terminal.

---

# 💡 Example Workflow

### 🔬 Research Mode

```text
Search Topic
     ↓
Retrieve 10 Research Papers
     ↓
Process Papers
     ↓
Build Retrieval Index
     ↓
Ask Questions
     ↓
Receive Grounded Answers
```

### 📄 PDF Mode

```text
Upload PDF
     ↓
Extract Text
     ↓
Chunk Document
     ↓
Create Embeddings
     ↓
FAISS + BM25
     ↓
CrossEncoder Reranking
     ↓
Ask Questions
```

### ▶️ YouTube Mode

```text
YouTube URL
     ↓
Extract Transcript
     ↓
Chunk Transcript
     ↓
Index Content
     ↓
Ask Questions / Summarize
```

---

# 🧪 Retrieval Pipeline in Detail

### Step 1 — Document Processing

Input documents are converted into text.

### Step 2 — Chunking

Large documents are divided into smaller overlapping chunks so that relevant passages can be retrieved efficiently.

### Step 3 — Embeddings

Each chunk is converted into a dense vector representation using a Sentence Transformer model.

### Step 4 — Semantic Search

FAISS performs vector similarity search to identify semantically related chunks.

### Step 5 — Keyword Search

BM25 retrieves chunks using lexical keyword matching.

### Step 6 — Hybrid Retrieval

Results from both retrieval approaches are combined.

### Step 7 — Reranking

A CrossEncoder evaluates the candidate chunks and reorders them according to relevance.

### Step 8 — Generation

The highest-ranked context is provided to the Groq-hosted LLM.

### Step 9 — Response

QueryMind produces a context-grounded answer or summary.

---

# 📊 Design Philosophy

QueryMind is built around a simple principle:

> **Retrieval quality matters before generation quality.**

Instead of immediately asking an LLM to answer from a large document, the system first focuses on finding the most useful pieces of information.

This creates a pipeline where:

**Better Retrieval → Better Context → Better Grounding**

---

# 🧩 What Makes QueryMind Different?

### 🔹 Multi-Source

Research papers, PDFs, and YouTube transcripts can all enter the same RAG architecture.

### 🔹 Hybrid Retrieval

Combines semantic similarity with keyword-based retrieval.

### 🔹 Reranking

Uses a CrossEncoder after initial retrieval to improve candidate ordering.

### 🔹 Grounded Generation

The LLM receives retrieved context instead of being asked to answer without supporting information.

### 🔹 Modular Architecture

The retrieval engine is separated from the Streamlit interface, making the system easier to test and extend.

### 🔹 Research-Oriented

The application is designed around research-paper discovery, document understanding, question answering, and summarization.

---

# 🔮 Future Improvements

Planned improvements include:

* [ ] Improve retrieval and ranking strategies
* [ ] Add systematic retrieval evaluation
* [ ] Expand RAG evaluation metrics
* [ ] Improve metadata handling
* [ ] Add citation-aware responses
* [ ] Add conversation memory
* [ ] Add multi-turn question answering
* [ ] Add persistent vector database support
* [ ] Optimize latency and token usage
* [ ] Deploy QueryMind as a production application

---

# 🔒 Security

API credentials should always be stored using environment variables or secure secret management.

Do **not** commit:

```text
.env
API keys
Access tokens
Private credentials
```

---

# 👩‍💻 Author

## Anusha Panicker

**B.Tech Artificial Intelligence & Machine Learning**

Aspiring AI Engineer focused on:

`Python` • `Machine Learning` • `RAG` • `LLMs` • `Information Retrieval`

---

<p align="center">

### 🧠 QueryMind

**Retrieve. Rerank. Understand.**

Built with Python • RAG • FAISS • BM25 • CrossEncoder • LLMs

</p>


