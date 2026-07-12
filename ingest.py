"""
DriveWise Ingestion Pipeline
────────────────────────────
Reads car-brochure PDFs → smart section-aware chunking →
FAISS vector index with rich metadata.

Filename convention:  Brand_Model.pdf  or  Brand_Model_v2.pdf
Examples:  Hyundai_Creta.pdf  |  Toyota_Fortuner_v2.pdf
"""

import os, json, pickle, re
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from config import (
    BROCHURE_DIR, VECTOR_STORE_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, SECTION_KEYWORDS
)


# ─────────────────────────── helpers ──────────────────────────────────────────

def detect_section(text: str) -> str:
    """Return the best-matching section label for a chunk of text."""
    text_lower = text.lower()
    best_section, best_score = "General", 0
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score, best_section = score, section
    return best_section


def chunk_text(text: str, page_num: int) -> list[dict]:
    """Split page text into overlapping word-level chunks with metadata."""
    words  = text.split()
    chunks = []
    step   = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text  = " ".join(chunk_words).strip()

        if len(chunk_text) < 60:          # skip near-empty chunks
            continue

        chunks.append({
            "text":        chunk_text,
            "page_number": page_num,
            "section":     detect_section(chunk_text),
            "word_count":  len(chunk_words),
        })

    return chunks


# ─────────────────────────── ingestion ────────────────────────────────────────

def ingest_brochure(pdf_path: str, brand: str, model: str,
                    doc_version: str = "v1") -> list[dict]:
    """Extract all text from a PDF and return enriched chunk dicts."""
    all_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text or len(text.strip()) < 40:
                continue

            page_chunks = chunk_text(text, page_num)
            for chunk in page_chunks:
                chunk.update({
                    "brand":       brand.strip().lower(),
                    "model":       model.strip().lower(),
                    "doc_version": doc_version,
                    "source_file": os.path.basename(pdf_path),
                    "total_pages": total_pages,
                })
            all_chunks.extend(page_chunks)

    print(f"  [{brand} {model}] {len(all_chunks)} chunks from {total_pages} pages")
    return all_chunks


# ─────────────────────────── vector store ─────────────────────────────────────

def build_vectorstore(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """Embed chunks and build a normalised FAISS inner-product index."""
    print(f"\nBuilding vectorstore for {len(chunks)} chunks …")
    embedder   = SentenceTransformer(EMBEDDING_MODEL)
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)                       # cosine similarity

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks                                 # metadata list mirrors index order


def save_vectorstore(index: faiss.Index, metadata: list[dict]) -> None:
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    with open(os.path.join(VECTOR_STORE_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"Vectorstore saved  →  {VECTOR_STORE_PATH}/")


def load_vectorstore() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    with open(os.path.join(VECTOR_STORE_PATH, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def _build_catalog(chunks: list[dict]) -> dict:
    catalog: dict[str, set] = {}
    for c in chunks:
        catalog.setdefault(c["brand"], set()).add(c["model"])
    return {k: sorted(v) for k, v in catalog.items()}


# ─────────────────────────── main entry point ─────────────────────────────────

def ingest_all_brochures(brochure_dir: str = BROCHURE_DIR) -> list[dict]:
    """
    Scan `brochure_dir` for PDFs named  Brand_Model[_vX].pdf,
    ingest them all, build and save the FAISS vectorstore.
    """
    os.makedirs(brochure_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(brochure_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{brochure_dir}'. Add files and run again.")
        return []

    all_chunks: list[dict] = []

    for pdf_file in pdf_files:
        stem  = os.path.splitext(pdf_file)[0]
        parts = stem.split("_")

        if len(parts) < 2:
            print(f"Skipping '{pdf_file}' – rename to Brand_Model.pdf")
            continue

        brand   = parts[0]
        model   = parts[1]
        version = parts[2] if len(parts) > 2 else "v1"

        print(f"\nIngesting: {brand} {model} (version={version})")
        chunks = ingest_brochure(
            os.path.join(brochure_dir, pdf_file), brand, model, version
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Nothing ingested – check file names.")
        return []

    index, metadata = build_vectorstore(all_chunks)
    save_vectorstore(index, metadata)

    catalog = _build_catalog(all_chunks)
    with open(os.path.join(VECTOR_STORE_PATH, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\n✅  Ingestion complete! {len(all_chunks)} chunks indexed.")
    print("Catalog:", json.dumps(catalog, indent=2))
    return all_chunks


if __name__ == "__main__":
    ingest_all_brochures()
