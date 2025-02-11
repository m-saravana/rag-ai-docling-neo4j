from typing import List, Dict, Any
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
import os

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model="phi",
            base_url="http://localhost:11434"
        )

    def load_documents(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        try:
            loader = DoclingLoader(file_path)
            documents = loader.load()
            
            structured_content = []
            for doc in documents:
                structured_content.append({
                    'content': doc.page_content,
                    'metadata': {
                        'source_document': file_path,
                        **doc.metadata
                    }
                })
            
            return structured_content
            
        except Exception as e:
            print(f"Error processing document: {e}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{
                    'content': f.read(),
                    'metadata': {
                        'source_document': file_path,
                        'content_type': 'text'
                    }
                }]

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_docs = []
    
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.embed_query(chunk)
                embedding = [float(x) for x in embedding]
                
                processed_docs.append({
                    'content': chunk,
                    'embedding': embedding,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': str(i), 
                        'chunk_size': str(len(chunk))
                    }
                })
        
        return processed_docs

    def extract_relationships(self, processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relationships = []
        
        for i, doc in enumerate(processed_docs):
            if i > 0:
                relationships.append({
                    'source': processed_docs[i-1]['metadata']['chunk_id'],
                    'target': doc['metadata']['chunk_id'],
                    'type': 'NEXT'
                })
            
            relationships.append({
                'source': doc['metadata']['chunk_id'],
                'target': doc['metadata']['source_document'],
                'type': 'FROM_DOCUMENT'
            })

        return relationships