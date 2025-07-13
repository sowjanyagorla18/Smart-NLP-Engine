from fastapi import FastAPI
import uvicorn

from app.api.routes_nlp import router as nlp_router
from app.rag.routes import router as rag_router

# Create the main FastAPI app
app = FastAPI(
    title="NLP and RAG API",
    description="A simple API for Natural Language Processing and Retrieval-Augmented Generation",
    version="1.0.0"
)


# Include the NLP routes with prefix
app.include_router(nlp_router, prefix="/nlp", tags=["NLP"])

# Include the RAG routes with prefix  
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

# Simple health check endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to NLP and RAG API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NLP and RAG API"}

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  
    ) 