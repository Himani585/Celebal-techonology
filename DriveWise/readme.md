
# DriveWise - Metadata Aware Automotive RAG Assistant

## Project Overview
An intelligent conversational assistant that helps users understand 
car brochures by answering natural language questions using a 
production-grade RAG pipeline.

## Features
- Metadata-filtered vector retrieval (FAISS)
- Cross-encoder re-ranking
- LLM answer generation (Groq)
- Source attribution
- 3-metric evaluation system
- SQLite query logging
- Analytics dashboard

## Tech Stack
- Python, Streamlit
- FAISS, Sentence Transformers
- Groq (llama-3.3-70b-versatile)
- LangChain, pdfplumber
- SQLite, Plotly

## Setup Instructions

### 1. Install dependencies
pip install -r requirements.txt

### 2. Get free Groq API key
https://console.groq.com/keys

### 3. Create .env file
GROQ_API_KEY=your_groq_api_key_here

### 4. Run the app
streamlit run app.py

### 5. Ingest a brochure
- Click Ingest Brochures tab
- Upload any car brochure PDF
- Enter brand and model name
- Click Ingest

### 6. Start chatting!
- Select brand and model
- Ask anything about the car

## Architecture
User Question
      ↓
Metadata Filter (brand + model)
      ↓
Vector Retrieval (FAISS)
      ↓
Cross-Encoder Re-Ranking
      ↓
LLM Generation (Groq)
      ↓
Answer + Sources + Evaluation

## Evaluation Metrics
- Faithfulness - Is answer grounded in context?
- Context Relevance - Are chunks relevant?
- Answer Completeness - Is question fully answered?

## Download Sample Brochure
https://www.hyundai.com/in/en/find-a-car/creta/highlights
