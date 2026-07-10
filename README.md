# Tech Layoffs RAG Assistant

An AI-powered question-answering system built on 6 years of tech layoff data (2020-2026). 
Ask natural language questions and get accurate, data-grounded answers with cited sources.

## What it does

Instead of manually exploring charts and dashboards, you can ask questions like:
- "Which industry had the most layoffs overall?"
- "What was the media sentiment during the 2023 layoff wave?"
- "How did AI companies compare to non-AI companies?"

The system retrieves relevant data, passes it to an LLM, and returns accurate answers 
grounded in real data — not generated from memory.

## How it works (RAG Architecture)

1. **Data Preparation** — 2,782 text chunks created from layoff events and news articles
2. **Embedding** — Each chunk converted to a vector using sentence-transformers
3. **Vector Store** — Stored in ChromaDB for semantic search
4. **Retrieval** — User question embedded and matched against stored chunks
5. **Generation** — Relevant chunks passed to GPT-4o-mini to generate grounded answers
6. **Interface** — Streamlit chat UI with conversation history and source citations

## Tech Stack

- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database** — ChromaDB
- **LLM** — OpenAI GPT-4o-mini
- **Framework** — LangChain concepts (manual implementation)
- **Frontend** — Streamlit
- **Data** — 2,470 layoff events + 306 news articles (2020-2026)

## Project Structure

rag-layoffs-assistant/
├── data/                          # Layoff events and news sentiment CSVs
├── notebooks/
│   ├── 01_build_vector_store.ipynb  # Data chunking and ChromaDB setup
│   └── 02_rag_pipeline.ipynb        # Retrieval and generation pipeline
├── app/
│   └── app.py                     # Streamlit chat interface
├── requirements.txt
└── README.md

## Key Findings from the Data

- **541,293 total layoffs** tracked across 1,903 companies
- **2023 was the worst year** with 170,324 layoffs — the "Great Tech Layoff"
- **Amazon led** all companies with 49,624 total layoffs
- **Non-AI companies** laid off significantly more (392,263) than AI companies (149,030)
- **Media sentiment** was predominantly negative — 150 negative articles vs 119 positive

## Setup

```bash
# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" > .env

# Build vector store (run once)
jupyter notebook notebooks/01_build_vector_store.ipynb

# Run the app
cd app
streamlit run app.py
```