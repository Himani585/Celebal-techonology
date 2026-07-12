"""
DriveWise RAG Evaluator
────────────────────────
Three evaluation dimensions:

  1. Faithfulness       – Is the answer grounded in the retrieved context?
                          (LLM-as-judge)
  2. Context Relevance  – Are the retrieved chunks relevant to the query?
                          (embedding cosine similarity)
  3. Answer Completeness– Does the answer address all parts of the query?
                          (LLM-as-judge)
"""

import re
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL


class RAGEvaluator:

    def __init__(self):
        self.llm      = Groq(api_key=GROQ_API_KEY)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    # ──────────────────────────────────────────────────────────────────────────
    # Individual metrics
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_score(self, prompt: str) -> float:
        """Ask the LLM for a 0-1 float. Fallback to 0.5 on parse error."""
        try:
            resp = self.llm.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            # accept "0.87", ".87", "1", "0"
            match = re.search(r"\d+\.?\d*", raw)
            if match:
                return min(max(float(match.group()), 0.0), 1.0)
        except Exception:
            pass
        return 0.5

    def faithfulness(self, answer: str, context_texts: list[str]) -> float:
        """
        Score how faithful the answer is to the given context.
        1.0 = every claim is directly supported; 0.0 = hallucinated.
        """
        if not context_texts or not answer:
            return 0.0

        context = " ".join(context_texts)[:3000]   # trim to avoid token overflow
        prompt  = f"""Rate how faithful the answer is to the provided context.
1.0 = every claim is directly supported by the context.
0.0 = the answer contains hallucinations or information not in the context.

Context:
{context}

Answer:
{answer}

Respond with ONLY a decimal number between 0 and 1."""
        return self._llm_score(prompt)

    def context_relevance(self, query: str, context_texts: list[str]) -> float:
        """
        Average cosine similarity between the query embedding and each chunk.
        Measures whether retrieval pulled in relevant sections.
        """
        if not context_texts:
            return 0.0

        q_emb  = self.embedder.encode([query],         normalize_embeddings=True)
        c_embs = self.embedder.encode(context_texts,   normalize_embeddings=True)
        sims   = np.dot(c_embs, q_emb.T).flatten()
        return float(np.mean(sims))

    def answer_completeness(self, query: str, answer: str) -> float:
        """
        Score how completely the answer addresses the user's question.
        """
        if not answer:
            return 0.0

        prompt = f"""Does the answer fully address the user's question?
1.0 = complete answer to all aspects of the question.
0.5 = partial answer.
0.0 = answer does not address the question at all.

Question: {query}
Answer: {answer}

Respond with ONLY a decimal number between 0 and 1."""
        return self._llm_score(prompt)

    # ──────────────────────────────────────────────────────────────────────────
    # Combined evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, query: str, answer: str, sources: list[dict]) -> dict:
        """
        Run all three metrics. Returns a dict with individual + overall score.

        `sources` should be the list returned by DriveWiseRAG.query().
        """
        context_texts = [s.get("snippet", "") for s in sources if s.get("snippet")]

        faith   = self.faithfulness(answer, context_texts)
        rel     = self.context_relevance(query, context_texts)
        comp    = self.answer_completeness(query, answer)
        overall = round((faith * 0.45 + rel * 0.30 + comp * 0.25), 3)

        return {
            "faithfulness":          round(faith,   3),
            "context_relevance":     round(rel,     3),
            "answer_completeness":   round(comp,    3),
            "overall":               overall,
        }
