"""
DriveWise RAG Pipeline
──────────────────────
Full pipeline:
  1. Metadata-filtered vector retrieval  (FAISS)
  2. Cross-encoder re-ranking
  3. Context-window control
  4. LLM answer generation  (Groq)
  5. Source attribution
"""

import time, json, os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from groq import Groq
from ingest import load_vectorstore
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVAL, TOP_K_RERANKED,
    VECTOR_STORE_PATH,
)


class DriveWiseRAG:
    """End-to-end metadata-aware RAG system for car brochures."""

    def __init__(self):
        print("⚙  Loading embedding model …")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("⚙  Loading cross-encoder re-ranker …")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

        print("⚙  Loading vectorstore …")
        self.index, self.metadata = load_vectorstore()

        self.llm = Groq(api_key=GROQ_API_KEY)
        print("✅  DriveWise RAG is ready!\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Catalog
    # ──────────────────────────────────────────────────────────────────────────

    def get_catalog(self) -> dict:
        path = os.path.join(VECTOR_STORE_PATH, "catalog.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 – Metadata-filtered retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, brand: str, model: str) -> list[dict]:
        """
        Embed query → search FAISS → post-filter by brand+model metadata.
        We over-fetch (TOP_K_RETRIEVAL × 6) to guarantee enough results after
        filtering, especially when the index holds multiple brands.
        """
        query_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        k_search  = min(TOP_K_RETRIEVAL * 6, len(self.metadata))
        scores, indices = self.index.search(query_emb, k_search)

        brand_lower = brand.strip().lower()
        model_lower = model.strip().lower()
        results: list[dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.metadata[idx]
            if (chunk["brand"] == brand_lower
                    and chunk["model"] == model_lower):
                results.append({
                    **chunk,
                    "retrieval_score": float(score),
                    "chunk_index":     int(idx),
                })
            if len(results) >= TOP_K_RETRIEVAL:
                break

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 – Cross-encoder re-ranking
    # ──────────────────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        pairs  = [(query, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:TOP_K_RERANKED]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – LLM generation with source attribution
    # ──────────────────────────────────────────────────────────────────────────

    def _generate(self, query: str, chunks: list[dict],
                  brand: str, model: str) -> tuple[str, list[dict]]:

        if not chunks:
            return (
                f"I couldn't find relevant information in the "
                f"{brand.title()} {model.title()} brochure for this query. "
                "Please rephrase or ask about a different aspect.",
                []
            )

        # Build context block (controlled window)
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            context_blocks.append(
                f"[CHUNK {i} | Section: {c.get('section','General')} "
                f"| Page {c.get('page_number','?')}]\n{c['text']}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = f"""You are DriveWise, an expert automotive assistant.
You help users understand the {brand.title()} {model.title()} by answering questions
based ONLY on the official brochure excerpts provided below.

RULES:
1. Base your answer SOLELY on the brochure context provided.
2. If the answer is not in the context, say exactly:
   "This information is not available in the {model.title()} brochure."
3. Use simple, clear language — the user may not be a car expert.
4. Quote precise numbers/units when available.
5. Do NOT guess, infer beyond the text, or mention other car models.
6. Keep answers concise — 3–5 sentences unless a list is clearer."""

        user_prompt = f"""BROCHURE CONTEXT:
{context}

USER QUESTION: {query}

Answer based only on the brochure context above:"""

        response = self.llm.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.05,
            max_tokens=800,
        )

        answer = response.choices[0].message.content.strip()

        sources = [
            {
                "chunk_number": i,
                "section":      c.get("section", "General"),
                "page_number":  c.get("page_number", "?"),
                "source_file":  c.get("source_file", ""),
                "doc_version":  c.get("doc_version", "v1"),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "snippet":      c["text"][:250] + "…" if len(c["text"]) > 250
                                else c["text"],
            }
            for i, c in enumerate(chunks, 1)
        ]

        return answer, sources

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def query(self, user_query: str, brand: str, model: str) -> dict:
        """
        Full pipeline: retrieve → rerank → generate.
        Returns a dict with answer, sources, and diagnostic stats.
        """
        t0 = time.time()

        retrieved = self._retrieve(user_query, brand, model)
        reranked  = self._rerank(user_query, retrieved)
        answer, sources = self._generate(user_query, reranked, brand, model)

        return {
            "query":           user_query,
            "brand":           brand,
            "model":           model,
            "answer":          answer,
            "sources":         sources,
            "retrieved_count": len(retrieved),
            "reranked_count":  len(reranked),
            "response_time":   round(time.time() - t0, 2),
        }
