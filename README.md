# 🚗 DriveWise — Metadata-Aware Automotive RAG Assistant

An intelligent conversational assistant that helps users understand car brochures
by answering natural language questions, powered by a production-grade RAG pipeline.

---

## Architecture

```
User Question
     │
     ▼
Metadata Filter  ──────────────────────────────────────────────────────────────
(brand + model)                                                                │
     │                                                                         │
     ▼                                                                         │
Vector Retrieval   ← FAISS Index (sentence-transformers embeddings)            │
(top-12 chunks)                                                                │
     │                                                                         │
     ▼                                                                         │
Cross-Encoder Re-Ranking                                                       │
(top-4 chunks)                                                                 │
     │                                                                         │
     ▼                                                                         │
Context Window Control                                                         │
     │                                                                         │
     ▼                                                                         │
LLM Generation (Groq / llama3-70b) ◄──── System prompt with strict grounding │
     │                                                                         │
     ▼                                                                         │
Answer  +  Source Attribution  +  Evaluation Scores  +  Query Logging ────────┘
```

---

## Quick Start

### 1. Clone & install
```bash
git clone <repo>
cd drivewise
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env → add your GROQ_API_KEY
# Free key at: https://console.groq.com
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Ingest brochures
- Click **📥 Ingest Brochures** in the sidebar
- Upload any car brochure PDF (e.g. Hyundai Creta brochure from Hyundai India's website)
- Enter brand + model name → click **Ingest**
- Switch to **💬 Chat** and start asking!

---

## File Structure

```
drivewise/
├── app.py            # Streamlit UI  (Chat / Analytics / Ingest)
├── config.py         # All configuration in one place
├── ingest.py         # PDF → chunks → FAISS index
├── rag_pipeline.py   # Retrieve → Re-rank → Generate
├── evaluator.py      # Faithfulness / Context Relevance / Completeness
├── logger.py         # SQLite query logging
├── requirements.txt
├── .env.example
└── brochures/        # Drop PDFs here (or upload via UI)
```

---

## Key Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `all-MiniLM-L6-v2` | Fast, accurate, runs locally |
| Vector store | FAISS (IndexFlatIP) | No server needed, cosine similarity |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Significant precision boost over bi-encoder alone |
| LLM | Groq llama3-70b | Free tier, very fast inference |
| Chunking | Section-aware word-level | Preserves semantic coherence vs. fixed-size |
| Metadata filtering | Brand + model pre-filter | Prevents cross-car contamination |

---

## Evaluation Metrics

- **Faithfulness** — LLM-as-judge: is the answer grounded in the retrieved context?
- **Context Relevance** — Cosine similarity between query and retrieved chunks
- **Answer Completeness** — LLM-as-judge: does the answer address the full question?
- **Overall** — Weighted: 45% Faithfulness + 30% Relevance + 25% Completeness

---

## Brochure Sources

Download official brochures directly from manufacturer websites:
- Hyundai India: https://www.hyundai.com/in/en/find-a-car
- Maruti Suzuki: https://www.marutisuzuki.com
- Tata Motors: https://www.tatamotors.com
- Toyota India: https://www.toyotabharat.com
