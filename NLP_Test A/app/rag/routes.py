from fastapi import APIRouter, HTTPException
from app.rag.schemas import AddDocumentsRequest, QueryRequest
from app.rag.document_ingestion import document_ingestion_service
from app.rag.retrieval_service import retrieval_service

router = APIRouter()

@router.post("/documents/add")
async def add_documents_to_knowledge_base(request: AddDocumentsRequest):
    documents = [{"id": doc.id, "text": doc.text} for doc in request.documents]
    return await document_ingestion_service.add_documents_to_knowledge_base(documents)

@router.get("/documents/list")
async def list_documents():
    doc_ids = document_ingestion_service.list_documents()
    return {"document_ids": doc_ids}


