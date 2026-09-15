from engine import search_topic, build_corpus_from_texts, ask_question

print("=" * 50)
print("TEST: Topic search WITH API key")
print("=" * 50)
papers = search_topic("transformers neural networks", max_results=10)
print(f"Found {len(papers)} papers:")
for p in papers:
    print(" -", p["title"], f"({p['published']})")

if papers:
    print("\n" + "=" * 50)
    print("TEST: Q&A on searched papers")
    print("=" * 50)
    texts_with_sources = [(p["full_text"], p["title"]) for p in papers]
    corpus = build_corpus_from_texts(texts_with_sources)
    answer, sources = ask_question(corpus, "What are transformers used for?")
    print("ANSWER:", answer)
    print("SOURCES:", sources)