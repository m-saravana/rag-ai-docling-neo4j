from typing import List, Dict, Any
from neo4j import GraphDatabase
import numpy as np
import logging
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def store_documents(self, documents: List[Dict[str, Any]]):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            for i, doc in enumerate(documents):
                try:
                    embedding = [float(x) for x in doc['embedding']]
                    
                    metadata = doc['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    properties = {
                        'content': str(doc['content']),
                        'embedding': embedding,
                        'chunk_id': str(metadata.get('chunk_id', '')),
                        'chunk_size': str(metadata.get('chunk_size', '')),
                        'source_document': str(metadata.get('source_document', '')),
                        'source': str(metadata.get('source', ''))
                    }
                    
                    query = """
                    CREATE (c:Chunk {
                        content: $content,
                        embedding: $embedding,
                        chunk_id: $chunk_id,
                        chunk_size: $chunk_size,
                        source_document: $source_document,
                        source: $source
                    })
                    """
                    
                    session.run(query, **properties)
                    
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    logger.error(f"Document content: {doc.keys()}")
                    logger.error(f"Metadata type: {type(doc['metadata'])}")
                    logger.error(f"Metadata content: {doc['metadata']}")
                    raise

    def store_relationships(self, relationships: List[Dict[str, Any]]):
        """Store relationships between chunks"""
        with self.driver.session() as session:
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Chunk)
                    WHERE a.metadata.chunk_id = $source
                    MATCH (b:Chunk)
                    WHERE b.metadata.chunk_id = $target
                    CREATE (a)-[r:$type]->(b)
                    """,
                    source=str(rel['source']),
                    target=str(rel['target']),
                    type=rel['type']
                )

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = [float(x) for x in query_embedding]
        logger.debug(f"Searching with query embedding of size: {len(query_embedding)}")
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                RETURN 
                    c.content as content,
                    c.embedding as embedding,
                    c.chunk_id as chunk_id,
                    c.source_document as source_document,
                    c.chunk_size as chunk_size
                """
            )
            
            chunks = []
            for record in result:
                embedding = record['embedding']
                if embedding:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        [float(x) for x in embedding]
                    )
                    chunk = {
                        'content': record['content'],
                        'metadata': {
                            'chunk_id': record['chunk_id'],
                            'source_document': record['source_document'],
                            'chunk_size': record['chunk_size']
                        },
                        'similarity': float(similarity)
                    }
                    chunks.append(chunk)
                    logger.debug(f"Chunk {record['chunk_id']} similarity: {similarity:.4f}")
                    logger.debug(f"Content preview: {record['content'][:100]}...")
            
            chunks.sort(key=lambda x: x['similarity'], reverse=True)
            top_chunks = chunks[:top_k]
            
            logger.debug("\nTop matches:")
            for i, chunk in enumerate(top_chunks):
                logger.debug(f"{i+1}. Score: {chunk['similarity']:.4f}")
                logger.debug(f"   Source: {chunk['metadata']['source_document']}")
                logger.debug(f"   Content: {chunk['content'][:100]}...")
            
            return top_chunks

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def close(self):
        self.driver.close()