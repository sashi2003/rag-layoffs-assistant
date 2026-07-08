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
    chroma_client = chromadb.PersistentClient(path="../chroma_db")
    collection = chroma_client.get_collection("layoffs_rag")
    model = SentenceTransformer('all-MiniLM-L6-v2')
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