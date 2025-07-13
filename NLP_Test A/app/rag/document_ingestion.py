import chromadb
from typing import List, Dict, Optional
from dataclasses import dataclass
from app.rag.embedding_client import get_embeddings
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    text: str
    embedding: List[float]
    

class DocumentIngestionService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.documents = {} 
        self.document_ids = [] 

    async def add_documents_to_knowledge_base(self, documents: List[Dict[str, str]], metadata: Optional[List[Dict]] = None):
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        embedding_response = await get_embeddings(texts)
        
        # Extract embeddings from the correct nested structure
        embeddings = [emb['embedding'] for emb in embedding_response.get('result', {}).get('data', [])]
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Check each embedding
        for i, emb in enumerate(embeddings):
            if not emb:
                logger.warning(f"Empty embedding for document {i}!")
        
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=doc_ids
            )
            logger.info("Successfully added documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {str(e)}")
            raise
        
        return {
            "message": f"Successfully added {len(documents)} documents to ChromaDB",
            "document_count": len(documents),
            "total_documents": len(self.list_documents()),
            "embeddings_generated": len(embeddings)
        }

    def list_documents(self):
        results = self.collection.get()
        return results['ids'] if results and 'ids' in results else []

    def search_similar_documents(self, query_embedding: List[float], top_k: int = 5):
        logger.debug(f"Searching for similar documents with top_k={top_k}")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            similar_docs = []
            ids = results.get('ids') if results else None
            documents = results.get('documents') if results else None
            embeddings = results.get('embeddings') if results else None
            
            # Don't require embeddings to be present (they can be None)
            if ids and documents and len(ids) > 0 and len(ids[0]) > 0:
                for i, doc_id in enumerate(ids[0]):
                    # Handle case where embeddings might be None
                    embedding = []
                    if embeddings and embeddings[0] and i < len(embeddings[0]) and embeddings[0][i] is not None:
                        emb = embeddings[0][i]
                        embedding = [float(x) for x in emb] if isinstance(emb, (list, tuple)) else []
                    
                    doc = Document(
                        id=doc_id,
                        text=documents[0][i],
                        embedding=embedding,
                    )
                    similar_docs.append(doc)
                    logger.debug(f"Found similar document: ID={doc_id}")
            else:
                logger.warning("No documents found in ChromaDB query results")
                
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

document_ingestion_service = DocumentIngestionService()