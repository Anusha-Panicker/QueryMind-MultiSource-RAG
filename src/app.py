import html

import streamlit as st
from engine import (
    search_topic,
    read_uploaded_pdf,
    get_youtube_transcript,
    build_corpus_from_texts,
    ask_question,
    summarize_text
)

st.set_page_config(
    page_title="QueryMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom styling ----------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7F5AF0, #2CB67D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero p {
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }
    .card {
        background-color: #1a1c24;
        border: 1px solid #2a2d3a;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .answer-box {
        background: linear-gradient(135deg, rgba(127,90,240,0.08), rgba(44,182,125,0.08));
        border-left: 4px solid #7F5AF0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .source-pill {
        display: inline-block;
        background-color: #2a2d3a;
        color: #2CB67D;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7F5AF0, #2CB67D);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .youtube-panel {
        background: linear-gradient(135deg, #17212b 0%, #102d32 100%);
        border: 1px solid #28515a;
        border-radius: 12px;
        padding: 1.25rem 1.4rem 0.9rem;
        margin-bottom: 1rem;
    }
    .youtube-kicker {
        color: #2CB67D;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .youtube-title {
        color: #F5F7FA;
        font-size: 1.35rem;
        font-weight: 750;
        margin: 0;
    }
    .youtube-copy {
        color: #B8C4CC;
        margin: 0.35rem 0 0.9rem;
    }
    .youtube-url {
        background: #101820;
        border: 1px solid #35636b;
        border-radius: 8px;
        color: #D9F7EE;
        padding: 0.55rem 0.8rem;
        overflow-wrap: anywhere;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Hero header ----------------
st.markdown("""
<div class="hero">
    <h1>🧠 QueryMind</h1>
    <p>Search research, upload PDFs, or drop a YouTube link — ask anything, get grounded answers.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Session state ----------------
if "corpus" not in st.session_state:
    st.session_state.corpus = None
    st.session_state.corpus_label = None
    st.session_state.raw_text_for_summary = None
    st.session_state.paper_list = None

# ---------------- Tabs for source selection ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Search a Topic", "📄 Upload a PDF", "▶️ YouTube Video"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    topic = st.text_input("Enter any research topic", placeholder="e.g. transformers, reinforcement learning, protein folding")
    col1, col2 = st.columns([1, 3])
    with col1:
        num_papers = st.slider("Papers to fetch", 5, 20, 15)
    if st.button("🔎 Search Papers", key="search_btn"):
        with st.spinner(f"Searching for papers on '{topic}'..."):
            papers = search_topic(topic, max_results=num_papers)
            if papers:
                texts_with_sources = [(p["full_text"], p["title"]) for p in papers]
                st.session_state.corpus = build_corpus_from_texts(texts_with_sources)
                st.session_state.corpus_label = f"topic:{topic}"
                st.session_state.paper_list = papers
                st.session_state.raw_text_for_summary = None
            else:
                st.error("No papers found with accessible PDFs. Try a different topic.")
    if st.session_state.paper_list and st.session_state.corpus_label and st.session_state.corpus_label.startswith("topic:"):
        st.success(f"✅ {len(st.session_state.paper_list)} papers loaded, {len(st.session_state.corpus['all_chunks'])} chunks indexed")
        with st.expander("📚 View all papers"):
            for p in st.session_state.paper_list:
                st.markdown(f"**{p['title']}**  \n*{', '.join(p['authors'][:3])} — {p['published']}*")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("📥 Process PDF", key="pdf_btn"):
        with st.spinner("Reading and indexing your PDF..."):
            text = read_uploaded_pdf(uploaded_file)
            st.session_state.corpus = build_corpus_from_texts([(text, uploaded_file.name)])
            st.session_state.corpus_label = f"pdf:{uploaded_file.name}"
            st.session_state.raw_text_for_summary = text
            st.session_state.paper_list = None
    if st.session_state.corpus_label and st.session_state.corpus_label.startswith("pdf:"):
        st.success(f"✅ Loaded — {len(st.session_state.corpus['all_chunks'])} chunks indexed")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="youtube-panel">'
        '<div class="youtube-kicker">Video intelligence</div>'
        '<p class="youtube-title">Bring a conversation into your research workspace</p>'
        '<p class="youtube-copy">Paste a public YouTube link. Hindi and English transcripts are supported.</p>'
        '</div>',
        unsafe_allow_html=True
    )
    yt_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Use a standard YouTube, youtu.be, or Shorts URL with captions available."
    )
    if st.button("▶️ Load transcript", key="yt_btn", use_container_width=True):
        if not yt_url.strip():
            st.warning("Paste a YouTube URL to continue.")
        else:
            with st.spinner("Finding Hindi or English captions and indexing the video..."):
                try:
                    transcript = get_youtube_transcript(yt_url.strip())
                    st.session_state.corpus = build_corpus_from_texts([(transcript, yt_url.strip())])
                    st.session_state.corpus_label = f"youtube:{yt_url.strip()}"
                    st.session_state.raw_text_for_summary = transcript
                    st.session_state.paper_list = None
                except Exception as e:
                    st.error(f"Couldn't fetch transcript: {e}")
    if st.session_state.corpus_label and st.session_state.corpus_label.startswith("youtube:"):
        st.success("✅ Transcript loaded and indexed. Hindi and English captions are supported.")
        st.markdown(
            f'<div class="youtube-url">🔗 {html.escape(st.session_state.corpus_label[len("youtube:") :])}</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Q&A + Summarize section ----------------
if st.session_state.corpus is not None:
    st.markdown("---")
    question = st.text_input("💬 Ask a question", placeholder="What is this about?")
    if st.button("Ask", key="ask_btn") and question:
        with st.spinner("Thinking..."):
            answer, sources = ask_question(st.session_state.corpus, question)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        unique_sources = list(dict.fromkeys(sources))
        st.markdown("**Sources:**")
        pills = "".join([f'<span class="source-pill">📄 {s}</span>' for s in unique_sources])
        st.markdown(pills, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    if st.button("📝 Summarize", key="summarize_btn"):
        with st.spinner("Summarizing..."):
            if st.session_state.raw_text_for_summary:
                text_to_summarize = st.session_state.raw_text_for_summary
                label = st.session_state.corpus_label
            else:
                all_texts = " ".join(st.session_state.corpus["all_chunks"][:30])
                text_to_summarize = all_texts
                label = "the loaded content"
            summary = summarize_text(text_to_summarize, source_name=label)
            st.session_state.last_summary = summary
    if "last_summary" in st.session_state:
        st.markdown(f'<div class="answer-box">{st.session_state.last_summary}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Choose a source above to get started — search a topic, upload a PDF, or paste a YouTube link.")