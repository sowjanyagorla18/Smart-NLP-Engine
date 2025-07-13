import httpx
from app.config import USF_API_URL, EMBEDDING_HEADERS
import json
import logging

logger = logging.getLogger(__name__)

async def get_embeddings(texts, model="usf1-embed"):
    if isinstance(texts, str):
        texts = [texts]
    
    payload = {"model": model, "input": texts}
    
    logger.debug(f"Requesting embeddings for {len(texts)} texts")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{USF_API_URL}/hiring/embed/embeddings", json=payload, headers=EMBEDDING_HEADERS)
            
            response_json = response.json()
            
            # Check if embeddings are empty
            if 'result' in response_json and 'data' in response_json['result'] and len(response_json['result']['data']) > 0:
                first_embedding = response_json['result']['data'][0].get('embedding', [])
                logger.debug(f"Generated embeddings successfully, first embedding length: {len(first_embedding)}")
            else:
                logger.warning("No embeddings found in API response")
            
            return response_json
            
        except Exception as e:
            logger.error(f"Error in embedding API call: {str(e)}")
            raise

async def rerank_texts(query, texts, model="usf1-rerank"):
    payload = {"model": model, "query": query, "texts": texts}
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{USF_API_URL}/hiring/embed/reranker", json=payload, headers=EMBEDDING_HEADERS)
        return response.json() 