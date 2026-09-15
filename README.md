<div align="center">

<h1>🧠 QueryMind</h1>

<p>
<strong>AI-Powered Multi-Source Retrieval-Augmented Generation</strong>
</p>

<p>
Search research papers, upload PDFs, process YouTube transcripts, ask questions, and generate grounded summaries.
</p>

</div>

---

<h2>🚀 Overview</h2>

<p>
<strong>QueryMind</strong> is an end-to-end Retrieval-Augmented Generation (RAG) application designed to retrieve relevant information from multiple sources and generate context-grounded responses using a Large Language Model.
</p>

<p>
The system combines <strong>semantic retrieval, keyword-based retrieval, hybrid ranking, and CrossEncoder reranking</strong> before passing the most relevant context to the LLM.
</p>

<p>
QueryMind supports three input sources:
</p>

<ul>
<li>🔎 Research papers</li>
<li>📄 Uploaded PDF documents</li>
<li>▶️ YouTube video transcripts</li>
</ul>

<p>
The application is built with a <strong>Streamlit interface</strong> that allows users to interact with the complete RAG pipeline through a simple web interface.
</p>

<h2>🧠 Architecture</h2>

<pre>
                         User Input
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
        Research Papers     PDF          YouTube Video
             │               │               │
             ▼               ▼               ▼
        Text Extraction / Transcript Extraction
             │               │               │
             └───────────────┼───────────────┘
                             │
                             ▼
                       Text Chunking
                             │
                             ▼
                  Sentence Transformer
                       Embeddings
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
             FAISS Search            BM25 Search
          Semantic Retrieval       Keyword Retrieval
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                      Hybrid Retrieval
                             │
                             ▼
                    CrossEncoder
                       Reranking
                             │
                             ▼
                     Relevant Context
                             │
                             ▼
                        Groq LLM
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
              Q&A Answer           Summary
</pre>

<h2>✨ Features</h2>

<ul>
<li>🔎 Search a research topic and retrieve up to 10 relevant research papers</li>
<li>📄 PDF document upload and text extraction</li>
<li>▶️ YouTube transcript extraction</li>
<li>✂️ Recursive text chunking with configurable overlap</li>
<li>🧠 Sentence Transformer-based embeddings</li>
<li>⚡ FAISS semantic similarity search</li>
<li>🔤 BM25 keyword-based retrieval</li>
<li>🔀 Hybrid retrieval combining semantic and keyword search</li>
<li>🎯 CrossEncoder-based reranking</li>
<li>🤖 Groq LLM-based answer generation</li>
<li>🛡️ Context-grounded responses to reduce unsupported answers</li>
<li>🔍 Cosine-similarity based vector retrieval</li>
<li>🔄 Handling of comparison-based questions</li>
<li>📚 Special handling for small documents</li>
<li>📝 Content summarization</li>
<li>🖥️ Interactive Streamlit web interface</li>
</ul>

<h2>🔬 Retrieval Pipeline</h2>

<p>
QueryMind uses a multi-stage retrieval pipeline to improve the relevance of the information provided to the LLM.
</p>

<ol>
<li>
<strong>Text Chunking:</strong>
Documents are divided into smaller overlapping chunks using a recursive text splitter.
</li>

<li>
<strong>Semantic Embeddings:</strong>
Each chunk is converted into a dense vector representation using a Sentence Transformer model.
</li>

<li>
<strong>Semantic Retrieval:</strong>
FAISS performs vector similarity search to retrieve semantically relevant chunks.
</li>

<li>
<strong>Keyword Retrieval:</strong>
BM25 retrieves chunks based on important keyword matches.
</li>

<li>
<strong>Hybrid Retrieval:</strong>
Results from semantic and keyword retrieval are combined to improve retrieval coverage.
</li>

<li>
<strong>Reranking:</strong>
A CrossEncoder evaluates the retrieved candidates and reorders them according to relevance.
</li>

<li>
<strong>Context-Grounded Generation:</strong>
The highest-ranked chunks are provided to the Groq-hosted LLM to generate the final response.
</li>
</ol>

<h2>📚 Supported Sources</h2>

<table>
<tr>
<th>Source</th>
<th>Processing</th>
</tr>

<tr>
<td>🔎 Research Papers</td>
<td>Topic Search → Retrieve 10 Papers → Download PDFs → Extract Text → Index</td>
</tr>

<tr>
<td>📄 PDF Documents</td>
<td>Upload → Extract Text → Chunk → Index</td>
</tr>

<tr>
<td>▶️ YouTube Videos</td>
<td>URL → Transcript → Chunk → Index</td>
</tr>
</table>

<h2>🤖 Grounded Question Answering</h2>

<p>
After indexing a source, users can ask questions through the QueryMind interface.
</p>

<p>
The system retrieves relevant chunks before generating the response. The LLM is instructed to answer using the provided context rather than relying on unsupported information.
</p>

<p>
If the required information is not available in the retrieved context, the system is designed to avoid confidently generating an unsupported answer.
</p>

<h2>📝 Summarization</h2>

<p>
QueryMind also provides a summarization workflow for loaded content.
</p>

<ul>
<li>📄 Summarize uploaded PDF documents</li>
<li>▶️ Summarize YouTube transcript content</li>
<li>🔎 Summarize retrieved research-paper content</li>
</ul>

<p>
The summarization feature uses the same Groq-based LLM integration to generate a concise representation of the loaded content.
</p>

<h2>🛠️ Tech Stack</h2>

<table>
<tr>
<th>Technology</th>
<th>Purpose</th>
</tr>

<tr>
<td>Python</td>
<td>Core implementation</td>
</tr>

<tr>
<td>Streamlit</td>
<td>Interactive web interface</td>
</tr>

<tr>
<td>PyMuPDF</td>
<td>PDF text extraction</td>
</tr>

<tr>
<td>Sentence Transformers</td>
<td>Text embeddings</td>
</tr>

<tr>
<td>FAISS</td>
<td>Semantic vector retrieval</td>
</tr>

<tr>
<td>BM25</td>
<td>Keyword-based retrieval</td>
</tr>

<tr>
<td>CrossEncoder</td>
<td>Document reranking</td>
</tr>

<tr>
<td>Groq</td>
<td>LLM-based generation</td>
</tr>

<tr>
<td>LangChain</td>
<td>Text splitting and LLM integration</td>
</tr>

<tr>
<td>Semantic Scholar / arXiv</td>
<td>Research paper retrieval</td>
</tr>

<tr>
<td>YouTube Transcript API</td>
<td>Video transcript extraction</td>
</tr>
</table>

<h2>📁 Project Structure</h2>

<pre>
RAG-Research-Paper-QA/
│
├── .gitignore
├── requirements.txt
│
└── src/
    ├── app.py
    ├── engine.py
    └── test_engine.py
</pre>

<h2>⚙️ Installation</h2>

<h3>1. Clone the repository</h3>

<p>
Clone the QueryMind repository from GitHub and open the project directory.
</p>

<h3>2. Create a virtual environment</h3>

<pre>
python -m venv venv
</pre>

<h3>3. Activate the environment</h3>

<p><strong>Windows:</strong></p>

<pre>
venv\Scripts\activate
</pre>

<p><strong>macOS / Linux:</strong></p>

<pre>
source venv/bin/activate
</pre>

<h3>4. Install dependencies</h3>

<pre>
pip install -r requirements.txt
</pre>

<h2>🔐 Environment Variables</h2>

<p>
QueryMind uses environment variables for API credentials. Create a <code>.env</code> file in the project directory.
</p>

<pre>
GROQ_API_KEY=your_groq_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
</pre>

<p>
The <code>.env</code> file is excluded from Git using <code>.gitignore</code> and should never be committed to the repository.
</p>

<h2>▶️ Run the Application</h2>

<pre>
streamlit run src/app.py
</pre>

<p>
After running the command, open the Streamlit application in your browser.
</p>

<h2>💡 Example Workflow</h2>

<ol>
<li>Open QueryMind.</li>
<li>Select <strong>Search a Topic</strong> to retrieve <strong> 10 relevant research papers</strong>, or use <strong>Upload a PDF</strong> or <strong>YouTube Video</strong>.</li>
<li>Load the required content.</li>
<li>QueryMind extracts and processes the content.</li>
<li>The content is chunked and indexed.</li>
<li>Ask a question about the loaded content.</li>
<li>The system retrieves and reranks relevant information.</li>
<li>The Groq LLM generates a context-grounded answer.</li>
<li>Use the <strong>Summarize</strong> option when a concise overview is required.</li>
</ol>

<h2>🎯 Project Highlights</h2>

<ul>
<li>
<strong>Multi-source RAG:</strong>
Works with research papers, PDFs, and YouTube transcripts.
</li>

<li>
<strong>Hybrid Retrieval:</strong>
Combines semantic and keyword-based retrieval instead of relying on a single retrieval method.
</li>

<li>
<strong>Reranking:</strong>
Uses a CrossEncoder to improve the ordering of retrieved candidates.
</li>

<li>
<strong>Grounded Generation:</strong>
Provides retrieved context to the LLM to reduce unsupported responses.
</li>

<li>
<strong>Modular Design:</strong>
The retrieval and generation pipeline is separated from the Streamlit interface.
</li>

<li>
<strong>Practical Interface:</strong>
The complete pipeline can be used through an interactive web application.
</li>
</ul>

<h2>🔮 Future Improvements</h2>

<ul>
<li>⬜ Improve retrieval quality and ranking strategies</li>
<li>⬜ Add larger and more systematic evaluation datasets</li>
<li>⬜ Add retrieval-specific evaluation metrics</li>
<li>⬜ Improve document metadata handling</li>
<li>⬜ Add citation-aware answers</li>
<li>⬜ Add conversation history and multi-turn Q&A</li>
<li>⬜ Add persistent vector database support</li>
<li>⬜ Optimize latency and token usage</li>
<li>⬜ Deploy QueryMind as a production application</li>
</ul>

<h2>⚠️ Project Status</h2>

<p>
<strong>🚧 Under Active Development</strong>
</p>

<p>
QueryMind is an ongoing project. The current version focuses on building a reliable multi-source RAG pipeline with hybrid retrieval, reranking, grounded generation, and an interactive Streamlit interface.
</p>

<h2>🔒 Security Note</h2>

<p>
API keys and other sensitive credentials should always be stored using environment variables or secure secret management.
</p>

<p>
<strong>Never commit API keys or other secrets to GitHub.</strong>
</p>

<h2>👩‍💻 Author</h2>

<p>
<strong>Anusha Panicker</strong>
</p>

<p>
B.Tech AI & ML Student<br>
Aspiring AI Engineer
</p>

<hr>

<div align="center">

<p>
<strong>🧠 QueryMind — Retrieve. Rerank. Understand.</strong>
</p>

<p>
Built with Python, RAG, FAISS, BM25, CrossEncoder and LLMs.
</p>

</div>
