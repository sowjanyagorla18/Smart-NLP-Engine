from fastapi import APIRouter, HTTPException
import uuid
import asyncio
from app.schemas.nlp_models import FlexibleTextRequest
from app.services.llm_client import call_llm_api
from app.services.payload_builder import build_llm_payload
from app.services.webhook_service import webhook_service
from app.rag.retrieval_service import retrieval_service
from typing import Union, List

router = APIRouter()

async def is_query_about_target_topics(text: str) -> bool:
    """
    Use the LLM to check if the text is about Deep Learning, Machine Learning, or Agentic AI.
    Returns True if yes, False otherwise.
    """
    try:
        check_prompt = (
            "Is the following text about Deep Learning, Machine Learning, or Agentic AI? "
            "Reply with only 'yes' or 'no'. Text: "
        )
        payload = build_llm_payload(check_prompt, text)
        response = await call_llm_api(payload, timeout=15.0)
        return response.strip().lower().startswith('yes')
    except Exception:
        # If LLM call fails, assume it's not about target topics
        return False

async def fetch_information_from_llm(text: str) -> str:
    """
    Fetch information from LLM when additional context is needed.
    This function generates content based on the user's text.
    """
    try:
        info_prompt = f"Provide comprehensive information about: {text}. Give detailed explanation with examples."
        payload = build_llm_payload(info_prompt, text)
        return await call_llm_api(payload, timeout=30.0)
    except Exception as e:
        # If LLM call fails, return the original text
        return f"Error fetching additional information: {str(e)}. Processing original text: {text}"

async def fetch_information_from_rag(text: str) -> str:
    """
    Fetch information from RAG when additional context is needed.
    This function retrieves relevant documents and generates content.
    """
    try:
        # Retrieve relevant documents
        relevant_docs = await retrieval_service._search_and_rerank(text, top_k=5)
        context = "\n\n".join(relevant_docs)
        
        # Generate comprehensive information based on retrieved documents
        rag_prompt = f"Based on the following context, provide comprehensive information about: {text}\n\nContext:\n{context}"
        payload = build_llm_payload(rag_prompt, text)
        return await call_llm_api(payload, timeout=30.0)
    except Exception as e:
        # If RAG fails, fallback to direct LLM
        return await fetch_information_from_llm(text)

async def get_text_for_processing(req: FlexibleTextRequest) -> Union[str, List[str]]:
    """
    Get text for processing. Simplified logic to reduce LLM calls.
    """
    # Check if text is provided
    has_text = (
        req.text and 
        ((isinstance(req.text, str) and req.text.strip()) or 
         (isinstance(req.text, list) and any(t.strip() for t in req.text)))
    )
    
    if not has_text:
        # No text provided, return empty or default message
        return "No text provided for processing."
    
    # Handle single string
    if isinstance(req.text, str):
        # Check if it's about target topics first
        if await is_query_about_target_topics(req.text):
            return await fetch_information_from_rag(req.text)
        else:
            # For non-target topics, fetch information from LLM to provide context
            return await fetch_information_from_llm(req.text)
    
    # Handle list of texts
    processed_texts = []
    if req.text is not None:
        for text in req.text:
            if await is_query_about_target_topics(text):
                processed_texts.append(await fetch_information_from_rag(text))
            else:
                # For non-target topics, fetch information from LLM
                processed_texts.append(await fetch_information_from_llm(text))
    
    return processed_texts

async def process_with_webhook(req: FlexibleTextRequest, task_type: str):
    """Process NLP task with webhook notifications if webhook_url is provided, with RAG for target topics."""
    task_id = req.task_id or str(uuid.uuid4())
    if req.webhook_url:
        await webhook_service.send_processing_notification(str(req.webhook_url), task_id)
    try:
        # Get text for processing
        text_to_process = await get_text_for_processing(req)
        
        # Determine the prompt for the task
        if task_type == "classify":
            prompt = "Classify the following text"
        elif task_type == "entities":
            prompt = ("Extract all named entities from the following text. Return the result as a table with two columns: 'Entity' and 'Type'. Each entity should be on a new line.")
        elif task_type == "summarize":
            prompt = "Summarize the following text"
        elif task_type == "sentiment":
            prompt = "Analyze the sentiment of the following text"
        else:
            prompt = "Process the following text"

        # Helper to run the RAG flow
        async def run_rag_flow(user_text: str):
            try:
                # Optionally, modify the query for the NLP task
                rag_query = f"{prompt}: {user_text}"
                # Retrieve top 5 relevant documents
                relevant_docs = await retrieval_service._search_and_rerank(rag_query, top_k=5)
                # Combine the query and documents for the LLM
                context = "\n\n".join(relevant_docs)
                final_prompt = (
                    f"{prompt} (with the following context):\nUser Query: {user_text}\n\nRelevant Documents:\n{context}"
                )
                return await call_llm_api(build_llm_payload(final_prompt, user_text), timeout=30.0)
            except Exception as e:
                # If RAG fails, fallback to direct processing
                return await call_llm_api(build_llm_payload(prompt, user_text), timeout=30.0)

        # Handle both single string and list of texts
        if isinstance(text_to_process, list):
            results = []
            for t in text_to_process:
                try:
                    # If the text already contains comprehensive information (from fetch_information_from_llm/rag)
                    # then process it directly without additional LLM calls
                    if len(t) > 200:  # If text is long, it's likely already processed
                        results.append(await call_llm_api(build_llm_payload(prompt, t), timeout=30.0))
                    else:
                        # Short text, might need additional processing
                        if await is_query_about_target_topics(t):
                            results.append(await run_rag_flow(t))
                        else:
                            results.append(await call_llm_api(build_llm_payload(prompt, t), timeout=30.0))
                except Exception as e:
                    results.append(f"Error processing text: {str(e)}")
        else:
            try:
                # If the text already contains comprehensive information (from fetch_information_from_llm/rag)
                # then process it directly without additional LLM calls
                if len(text_to_process) > 200:  # If text is long, it's likely already processed
                    results = await call_llm_api(build_llm_payload(prompt, text_to_process), timeout=30.0)
                else:
                    # Short text, might need additional processing
                    if await is_query_about_target_topics(text_to_process):
                        results = await run_rag_flow(text_to_process)
                    else:
                        results = await call_llm_api(build_llm_payload(prompt, text_to_process), timeout=30.0)
            except Exception as e:
                results = f"Error processing text: {str(e)}"

        if req.webhook_url:
            await webhook_service.send_completion_notification(str(req.webhook_url), task_id, results)
        return {
            "task_id": task_id,
            "status": "completed",
            "result": results
        }
    except Exception as e:
        if req.webhook_url:
            await webhook_service.send_error_notification(str(req.webhook_url), task_id, str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/classify")
async def classify_text(req: FlexibleTextRequest):
    return await process_with_webhook(req, "classify")

@router.post("/entities")
async def extract_entities(req: FlexibleTextRequest):
    return await process_with_webhook(req, "entities")

@router.post("/summarize")
async def summarize_text(req: FlexibleTextRequest):
    return await process_with_webhook(req, "summarize")

@router.post("/sentiment")
async def analyze_sentiment(req: FlexibleTextRequest):
    return await process_with_webhook(req, "sentiment")
