import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

st.set_page_config(
    page_title = "Tech Layoffs AI Assistant", 
    layout = "wide"
)

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Load models and vector store
@st.cache_resource
def load_resources():
    import pandas as pd
    
    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')
    chroma_path = os.path.join(base_path, 'chroma_db')
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    # Check if collection already exists
    existing = [c.name for c in chroma_client.list_collections()]
    
    if "layoffs_rag" in existing:
        collection = chroma_client.get_collection("layoffs_rag")
        return collection, model
    
    # Build vector store if it doesn't exist
    st.info("Building vector store for first time — this takes 2-3 minutes...")
    
    # Load data
    df_layoffs = pd.read_csv(os.path.join(data_path, 'layoffs_events.csv'))
    df_news = pd.read_csv(os.path.join(data_path, 'news_sentiment.csv'))
    df_layoffs['date'] = pd.to_datetime(df_layoffs['date']).dt.strftime('%B %d, %Y')
    
    # Create chunks
    def create_layoff_chunk(row):
        company = row['company'] if pd.notna(row['company']) else 'Unknown company'
        location = row['location'] if pd.notna(row['location']) else 'Unknown location'
        count = f"{int(row['layoff_count'])} employees" if pd.notna(row['layoff_count']) else 'an unknown number of employees'
        date = row['date'] if pd.notna(row['date']) else 'unknown date'
        pct = f"({row['pct_workforce']} of workforce)" if pd.notna(row['pct_workforce']) else ''
        industry = row['industry'] if pd.notna(row['industry']) else 'Unknown'
        country = row['country'] if pd.notna(row['country']) else 'Unknown'
        ai = "an AI company" if row['is_ai_company'] == True else "a non-AI company"
        return f"{company}, based in {location} ({country}), laid off {count} {pct} on {date}. They operate in the {industry} industry and are {ai}."

    def create_news_chunk(row):
        date = row['date'] if pd.notna(row['date']) else 'unknown date'
        title = row['title'] if pd.notna(row['title']) else 'Unknown title'
        source = row['source'] if pd.notna(row['source']) else 'Unknown source'
        sentiment = row['sentiment_cat'] if pd.notna(row['sentiment_cat']) else 'unknown'
        description = row['description'] if pd.notna(row['description']) else ''
        chunk = f"On {date}, {source} published: '{title}'. The sentiment of this article was {sentiment}."
        if description:
            chunk += f" Description: {description}"
        return chunk

    df_layoffs['chunk'] = df_layoffs.apply(create_layoff_chunk, axis=1)
    df_news['chunk'] = df_news.apply(create_news_chunk, axis=1)
    
    # Summary chunks
    industry_totals = df_layoffs.groupby('industry')['layoff_count'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    ).sort_values(ascending=False).head(10)
    industry_summary = "Summary of total layoffs by industry: " + ", ".join([f"{ind}: {int(count):,} employees" for ind, count in industry_totals.items()])
    
    company_totals = df_layoffs.groupby('company')['layoff_count'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    ).sort_values(ascending=False).head(10)
    company_summary = "Summary of top companies by total layoffs: " + ", ".join([f"{comp}: {int(count):,} employees" for comp, count in company_totals.items()])
    
    ai_totals = df_layoffs.groupby('is_ai_company')['layoff_count'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    )
    ai_summary = f"AI vs Non-AI company layoffs: AI companies laid off {int(ai_totals.get(True, 0)):,} employees total, Non-AI companies laid off {int(ai_totals.get(False, 0)):,} employees total."
    
    df_layoffs['year'] = pd.to_datetime(df_layoffs['date'], format='%B %d, %Y', errors='coerce').dt.year
    year_totals = df_layoffs.groupby('year')['layoff_count'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    ).sort_values(ascending=False)
    year_summary = "Summary of total layoffs by year: " + ", ".join([f"{int(year)}: {int(count):,} employees" for year, count in year_totals.items() if pd.notna(year)])
    
    country_totals = df_layoffs.groupby('country')['layoff_count'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').sum()
    ).sort_values(ascending=False).head(5)
    country_summary = "Summary of layoffs by country (top 5): " + ", ".join([f"{country}: {int(count):,} employees" for country, count in country_totals.items()])
    
    sentiment_summary = f"Media sentiment analysis: Out of 306 articles, 150 were negative, 119 positive, and 37 neutral. The average sentiment score during 2023 was -0.0397, indicating predominantly negative coverage."
    
    summary_chunks = [industry_summary, company_summary, ai_summary, year_summary, country_summary, sentiment_summary]
    
    # Combine all chunks
    layoff_chunks = df_layoffs['chunk'].tolist()
    news_chunks = df_news['chunk'].tolist()
    all_chunks = layoff_chunks + news_chunks + summary_chunks
    
    all_ids = [f"layoff_{i}" for i in range(len(layoff_chunks))] + \
              [f"news_{i}" for i in range(len(news_chunks))] + \
              [f"summary_{i}" for i in range(len(summary_chunks))]
    
    all_metadata = [{"source": "layoffs", "index": i} for i in range(len(layoff_chunks))] + \
                   [{"source": "news", "index": i} for i in range(len(news_chunks))] + \
                   [{"source": "summary", "index": i} for i in range(len(summary_chunks))]
    
    # Build collection
    collection = chroma_client.create_collection(
        name="layoffs_rag",
        metadata={"hnsw:space": "cosine"}
    )
    
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        embeddings = model.encode(batch_chunks).tolist()
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metadata
        )
    
    return collection, model
collection, model = load_resources()

# RAG functions
def retrieve_context(query, n_results=5, chat_history=None):
    if chat_history and len(chat_history) >= 2:
        last_user_message = chat_history[-2]['content']
        enriched_query = f"{last_user_message} {query}"
    else:
        enriched_query = query
    
    results = collection.query(query_texts=[enriched_query], n_results=n_results)
    return results['documents'][0], results['metadatas'][0]

def generate_answer(query, chunks):
    context = "\n\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""You are an expert analyst of tech industry layoff trends.
Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so honestly.
Always mention specific companies, numbers, and dates when available.

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful tech industry analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# UI
st.title("Tech Layoffs AI Assistant")
st.markdown("*Ask any question about tech layoffs from 2020-2026*")
st.markdown("---")

# Example questions
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("- Which industry had the most layoffs?")
with col2:
    st.markdown("- What happened during the 2023 layoff wave?")
with col3:
    st.markdown("- How did AI companies compare to non-AI?")

st.markdown("---")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask a question about tech layoffs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching through layoff data..."):
            chunks, metadatas = retrieve_context(prompt, chat_history=st.session_state.messages)
            answer = generate_answer(prompt, chunks)
        
        st.markdown(answer)
        
        with st.expander("📚 Sources used"):
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                st.markdown(f"**Source {i+1}** [{meta['source']}]")
                st.markdown(f"_{chunk[:200]}..._")
                st.markdown("---")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})