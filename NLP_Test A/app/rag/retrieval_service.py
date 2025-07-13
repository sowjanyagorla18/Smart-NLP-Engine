import logging
from typing import List, Optional
from app.rag.document_ingestion import document_ingestion_service, Document
from app.rag.embedding_client import get_embeddings, rerank_texts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.document_service = document_ingestion_service

    async def _similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for similar documents using ChromaDB.
        Returns a list of Document objects.
        """
        logger.debug(f"Performing similarity search for query: '{query}' with top_k={top_k}")
        
        # Check documents in ChromaDB instead of empty dictionary
        doc_list = self.document_service.list_documents()
        logger.debug(f"Available documents in ChromaDB: {doc_list}")
        
        if not doc_list:
            logger.warning("No documents in ChromaDB")
            return []
            
        logger.info("Getting query embedding...")
        query_embedding_response = await get_embeddings(query)
        # Extract embedding from the correct nested structure
        query_embedding = query_embedding_response['result']['data'][0]['embedding']
        logger.debug(f"Query embedding generated, length: {len(query_embedding)}")
        
        logger.info("Searching in ChromaDB...")
        similar_docs = self.document_service.search_similar_documents(query_embedding, top_k)
        logger.info(f"Similarity search completed, found {len(similar_docs)} documents")
        
        return similar_docs

    async def _search_and_rerank(self, query: str, top_k: int = 5):
        """
        Search for similar documents and rerank them.
        Returns a list of document texts (strings) only.
        """
        similar_docs = await self._similarity_search(query, top_k)
        if not similar_docs:
            return []
        texts = [doc.text for doc in similar_docs]
        logger.info("Reranking documents...")
        rerank_response = await rerank_texts(query, texts)
        # Extract reranked data from the correct nested structure
        reranked_data = rerank_response['result']['data']
        reranked_data.sort(key=lambda x: x['score'], reverse=True)
        return [item['text'] for item in reranked_data[:top_k]]

    async def process_nlp_task_with_rag(self, task_type: str, query: str, top_k: int = 5):
        """
        Main entry: Given a query, search similar docs, rerank, and return only the text of the top result.
        """
        doc_list = self.document_service.list_documents()
        if not doc_list:
            logger.info("No documents in knowledge base, processing without RAG")
            return ""
        logger.info("Searching for relevant documents...")
        relevant_docs = await self._search_and_rerank(query, top_k)
        if not relevant_docs:
            logger.info("No relevant documents found, processing without RAG")
            return ""
        logger.info("Returning top reranked document text...")
        return relevant_docs[0]  


retrieval_service = RetrievalService() 