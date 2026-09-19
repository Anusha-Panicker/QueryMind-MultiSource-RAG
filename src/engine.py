import os
import time
import re
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from groq import Groq
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-120b")


@lru_cache(maxsize=1)
def get_embed_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model on {device}...")
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)


@lru_cache(maxsize=1)
def get_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading reranker on {device}...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def search_topic_arxiv(topic, max_results=15):
    """Fallback source: arXiv, used if Semantic Scholar is rate-limited."""
    import arxiv
    import fitz

    search = arxiv.Search(
        query=f'all:"{topic}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    client_arxiv = arxiv.Client(page_size=10, delay_seconds=3, num_retries=2)

    results = list(client_arxiv.results(search))

    def load_result(item):
        result, index = item
        response = requests.get(result.pdf_url, timeout=15)
        doc = fitz.open(stream=response.content, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        return {
            "id": f"paper_{index + 1}",
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": str(result.published.date()),
            "pdf_url": result.pdf_url,
            "full_text": full_text
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded = executor.map(load_result, enumerate(results))
        papers = [paper for paper in loaded if paper["full_text"].strip()]
    return papers


# ---------------- SOURCE 1: Topic search (Semantic Scholar) ----------------

@lru_cache(maxsize=16)
def search_topic(topic, max_results=15, max_retries=3):
    """
    Tries Semantic Scholar first (broader coverage). If it's rate-limited
    even after retries, automatically falls back to arXiv.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": topic, "limit": min(max_results * 3, 50), "fields": "title,authors,year,openAccessPdf,url"}
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    results = None
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("data", [])
            break
        elif response.status_code == 429:
            wait_time = 10 * (attempt + 1)
            print(f"Semantic Scholar rate limited, waiting {wait_time}s (retry {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        else:
            break  # some other error, don't keep retrying

    if results is None:
        print("Semantic Scholar unavailable — falling back to arXiv...")
        try:
            return search_topic_arxiv(topic, max_results=max_results)
        except Exception as e:
            raise RuntimeError(
                f"Both Semantic Scholar and arXiv are currently rate-limited or unavailable. "
                f"Please wait a few minutes and try again. (arXiv error: {e})"
            )

    candidates = [
        result for result in results
        if result.get("openAccessPdf", {}).get("url")
    ][:max_results * 2]

    def load_result(item):
        result, index = item
        import fitz

        pdf_url = result["openAccessPdf"]["url"]
        try:
            response = requests.get(pdf_url, timeout=15)
            doc = fitz.open(stream=response.content, filetype="pdf")
            full_text = "".join(page.get_text("text", sort=True) for page in doc)
            doc.close()
        except Exception:
            return None
        if len(full_text.strip()) < 500:
            return None
        return {
            "id": f"paper_{index + 1}",
            "title": result.get("title"),
            "authors": [a.get("name") for a in result.get("authors", [])],
            "published": str(result.get("year")),
            "pdf_url": pdf_url,
            "full_text": full_text
        }

    with ThreadPoolExecutor(max_workers=6) as executor:
        papers = [paper for paper in executor.map(load_result, enumerate(candidates)) if paper]
    papers = papers[:max_results]

    if len(papers) == 0:
        print("Semantic Scholar returned no usable open-access PDFs — falling back to arXiv...")
        return search_topic_arxiv(topic, max_results=max_results)

    return papers


# ---------------- SOURCE 2: Uploaded PDF ----------------

def read_uploaded_pdf(uploaded_file):
    """Takes a Streamlit-uploaded file, returns its extracted text."""
    import fitz

    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "".join(page.get_text("text", sort=True) for page in doc)
    doc.close()
    return full_text


# ---------------- SOURCE 3: YouTube transcript ----------------

def extract_youtube_id(url):
    """Pulls the video ID out of common YouTube URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract a valid YouTube video ID from that URL.")


def get_youtube_transcript(url):
    """Return a Hindi or English YouTube transcript as plain text."""
    video_id = extract_youtube_id(url)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=("hi", "en"))
    full_text = " ".join([snippet.text for snippet in fetched_transcript])
    if not full_text.strip():
        raise ValueError("The video transcript is empty or unavailable.")
    return full_text


# ---------------- Building a searchable corpus from any source ----------------

def build_corpus_from_texts(texts_with_sources, chunk_size=800, chunk_overlap=100, existing_corpus=None):
    """
    texts_with_sources: list of (text, source_name) tuples.
    Builds chunks, embeddings, FAISS index, and BM25 index.
    If existing_corpus is provided, it updates it incrementally.
    """
    import faiss
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    new_chunks = []
    new_chunk_sources = []
    for text, source in texts_with_sources:
        for chunk in splitter.split_text(text):
            new_chunks.append(chunk)
            new_chunk_sources.append(source)

    if existing_corpus is None:
        all_chunks = new_chunks
        chunk_sources = new_chunk_sources

        embeddings = get_embed_model().encode(all_chunks, show_progress_bar=False)
        normalized_embeddings = normalize_vectors(np.array(embeddings).astype('float32'))
        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
    else:
        all_chunks = existing_corpus["all_chunks"] + new_chunks
        chunk_sources = existing_corpus["chunk_sources"] + new_chunk_sources
        index = existing_corpus["index"]

        # Incremental FAISS update
        new_embeddings = get_embed_model().encode(new_chunks, show_progress_bar=False)
        normalized_new_embeddings = normalize_vectors(np.array(new_embeddings).astype('float32'))
        index.add(normalized_new_embeddings)

        # Incremental BM25 update (re-initialize since BM25Okapi doesn't support add)
        tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

    return {
        "all_chunks": all_chunks,
        "chunk_sources": chunk_sources,
        "index": index,
        "bm25": bm25
    }


print("engine.py loaded successfully!")

# ---------------- Hybrid search + re-ranking (works on ANY corpus) ----------------

def hybrid_search(corpus, question, top_k=5, candidate_pool=15, max_per_source=2):
    all_chunks = corpus["all_chunks"]
    chunk_sources = corpus["chunk_sources"]
    index = corpus["index"]
    bm25 = corpus["bm25"]

    question_embedding = get_embed_model().encode([question])
    question_embedding = normalize_vectors(np.array(question_embedding).astype('float32'))
    candidate_pool = min(candidate_pool, len(all_chunks))
    distances, vector_indices = index.search(question_embedding, candidate_pool)
    vector_indices = vector_indices[0]

    tokenized_question = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_question)
    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]

    combined_scores = {}
    for rank, idx in enumerate(vector_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + 1 / (rank + 1)
    for rank, idx in enumerate(bm25_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + 1 / (rank + 1)

    sorted_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)
    final_indices, source_count = [], {}
    for idx in sorted_indices:
        src = chunk_sources[idx]
        if source_count.get(src, 0) < max_per_source:
            final_indices.append(idx)
            source_count[src] = source_count.get(src, 0) + 1
        if len(final_indices) >= top_k:
            break
    return final_indices


def hybrid_search_reranked(corpus, question, top_k=5, candidate_pool=15, max_per_source=3):
    all_chunks = corpus["all_chunks"]
    candidates = hybrid_search(corpus, question, top_k=candidate_pool, candidate_pool=candidate_pool, max_per_source=max_per_source)
    pairs = [[question, all_chunks[idx]] for idx in candidates]
    rerank_scores = get_reranker().predict(pairs)
    scored = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in scored[:top_k]]


def get_context_for_question(corpus, question, top_k=5, small_doc_threshold=25):
    all_chunks = corpus["all_chunks"]
    if len(all_chunks) <= small_doc_threshold:
        return list(range(len(all_chunks)))
    else:
        return hybrid_search_reranked(corpus, question, top_k=top_k)


def expand_query(question):
    """
    Uses the LLM to generate 3 semantically similar variations of the user query
    to improve retrieval recall.
    """
    prompt = (
        "You are an AI research assistant. Your task is to take a user question and generate 3 alternative "
        "versions of the question that capture the same intent but use different wording. This will help "
        "improve retrieval recall. Output only the 3 questions, one per line, without numbering."
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    variations = response.choices[0].message.content.strip().split("\n")
    # Clean up any numbering if the LLM ignored the "without numbering" instruction
    cleaned_variations = [re.sub(r"^\d+[\.\)]\s*", "", v).strip() for v in variations if v.strip()]
    return cleaned_variations[:3]

def ask_question(corpus, question, top_k=5, small_doc_threshold=25):
    all_chunks = corpus["all_chunks"]
    chunk_sources = corpus["chunk_sources"]

    if len(all_chunks) <= small_doc_threshold:
        top_indices = list(range(len(all_chunks)))
    else:
        # Multi-Query Expansion for better recall
        expanded_queries = expand_query(question)
        all_queries = [question] + expanded_queries

        all_indices = []
        for q in all_queries:
            indices = hybrid_search_reranked(corpus, q, top_k=top_k)
            all_indices.extend(indices)

        # Deduplicate while preserving order
        seen = set()
        top_indices = [i for i in all_indices if not (i in seen or seen.add(i))][:top_k]

    retrieved_chunks = [all_chunks[idx] for idx in top_indices]
    sources = [chunk_sources[idx] for idx in top_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = ("Answer the question in English using ONLY the context below. "
              "Translate relevant Hindi content into clear English. If the answer isn't in the context, say so.\n\n"
              "Context:\n" + context + "\n\nQuestion: " + question + "\n\nAnswer:")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, sources


# ---------------- Summarization (works on ANY raw text) ----------------

def summarize_text(text, source_name="this document", max_chars=15000):
    """
    Summarizes any block of text (a paper's full text, a PDF, or a YouTube
    transcript). If text is very long, only the first max_chars are used
    to stay within the LLM's context comfortably.
    """
    trimmed_text = text[:max_chars]

    prompt = (f"Summarize the following content from {source_name} in a clear, well-structured way. "
               "Include the main points, key findings or arguments, and any important conclusions. "
               "Use short paragraphs or bullet points where helpful.\n\n"
               "Content:\n" + trimmed_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


print("engine.py fully loaded — search, Q&A, and summarization ready!")