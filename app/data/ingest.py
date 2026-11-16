"""Ingestion pipeline: load -> chunk -> embed -> store (spec 5.8 A)."""
from pathlib import Path

from sqlalchemy.orm import Session

from app.agent.embeddings import EmbeddingProvider
from app.data import repository as repo

CORPUS_DIR = Path(__file__).parent / "corpus"

# WHY 150 words: all-MiniLM-L6-v2 silently truncates input at 256 TOKENS. ~1.3 tokens/word
# means 150 words ~ 200 tokens, safely under the limit, so no chunk text is dropped from the
# embedding (the same silent-truncation bug class as the Auditor project).
CHUNK_WORDS = 150
OVERLAP_WORDS = 25  # WHY overlap: a sentence cut at a boundary still appears whole in one chunk


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def ingest_text(s: Session, embedder: EmbeddingProvider, doc_id: str, title: str, text: str) -> int:
    chunks = chunk_text(text)
    vectors = embedder.embed(chunks)
    repo.add_doc_chunks(s, [
        {"doc_id": doc_id, "doc_title": title, "chunk_index": i, "content": c, "embedding": v,
         "metadata_json": {"source": "corpus"}}
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=False))
    ])
    return len(chunks)


def ingest_corpus(s: Session, embedder: EmbeddingProvider, corpus_dir: Path = CORPUS_DIR) -> int:
    repo.clear_doc_chunks(s)
    total = 0
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].strip()
        total += ingest_text(s, embedder, path.stem, title, text)
    return total
