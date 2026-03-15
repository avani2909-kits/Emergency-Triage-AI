import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# Page settings
st.set_page_config(
    page_title="Emergency Triage AI",
    page_icon="🚑",
    layout="wide"
)

st.title("🚑 Real-Time Emergency Triage AI Assistant")

st.success("⚡ Low-Latency AI Retrieval System for Emergency Protocols")

st.markdown(
"""
This AI assistant analyzes medical documents and recommends
the **next emergency action** based on patient symptoms.
"""
)

# Example symptoms
st.markdown("### Example Symptoms")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Chest pain + unconscious")

with col2:
    st.warning("Heavy bleeding after accident")

with col3:
    st.success("Slurred speech + face drooping")

# Upload PDF
uploaded_file = st.file_uploader(
    "📄 Upload Medical Protocol PDF",
    type="pdf"
)

docs = []

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            docs.append(text)

    st.success("Medical document loaded successfully!")

    # Load AI embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert documents to embeddings
    embeddings = model.encode(docs)

    embeddings = np.array(embeddings)

    # Create FAISS search index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # User query
    query = st.text_input("🩺 Enter Patient Symptoms")

    if query:

        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding)

        # Search most relevant protocol
        distances, indices = index.search(query_embedding, k=1)

        result = docs[indices[0][0]]

        st.subheader("🚨 Recommended Emergency Protocol")

        st.write(result)

else:
    st.info("Upload a medical PDF to start analysis.")
