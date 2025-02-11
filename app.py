import streamlit as st
from document_processor import DocumentProcessor
from graph_store import Neo4jStore
from llm import LLMProcessor
import os
from dotenv import load_dotenv
import json

load_dotenv()

@st.cache_resource
def init_components():
    os.environ['HF_HOME'] = './hf_cache'
    doc_processor = DocumentProcessor()
    graph_store = Neo4jStore(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    llm_processor = LLMProcessor()
    return doc_processor, graph_store, llm_processor

doc_processor, graph_store, llm_processor = init_components()

st.title("RAG Q&A System")

# File upload
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            with open(f"temp_{file.name}", "wb") as f:
                f.write(file.getvalue())
            
            documents = doc_processor.load_documents(f"temp_{file.name}")
            processed_docs = doc_processor.process_documents(documents)
            relationships = doc_processor.extract_relationships(processed_docs)
            
            for doc in processed_docs:
                if 'dl_meta' in doc:
                    doc['dl_meta'] = json.dumps(doc['dl_meta'])
            
            for doc in processed_docs:
                for key, value in doc.items():
                    if isinstance(value, dict):
                        doc[key] = json.dumps(value)
            
            
            # Store in Neo4j
            graph_store.store_documents(processed_docs)
            
            # Clean up
            os.remove(f"temp_{file.name}")
    
    st.success("Documents processed successfully!")


query = st.text_input("Ask a question about your documents")

if query:
    with st.spinner("Searching for answer..."):
        query_embedding = doc_processor.embedding_model.embed_query(query)
        
        similar_chunks = graph_store.similarity_search(query_embedding)

        response = llm_processor.generate_response(query, similar_chunks)
        
        st.write("Answer:", response)
        
        with st.expander("Source Chunks"):
            for chunk in similar_chunks:
                st.write(chunk['content'])
                st.write("---")