from pydantic import BaseModel
from typing import List, Dict, Optional

class DocumentInput(BaseModel):
    id: str
    text: str

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentInput]

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5 