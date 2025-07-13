# NLP 

A FastAPI-based Natural Language Processing (NLP) API with Retrieval-Augmented Generation (RAG) capabilities. This project provides intelligent text processing services including classification, entity extraction, summarization, and sentiment analysis, with automatic RAG integration for domain-specific queries.

## 🚀 Project Overview

This API combines traditional NLP tasks with modern RAG (Retrieval-Augmented Generation) technology to provide enhanced text processing capabilities. The system intelligently detects domain-specific queries (Deep Learning, Machine Learning, Agentic AI) and automatically applies RAG processing for more accurate and contextual responses.

### Key Features

- **Multi-task NLP Processing**: Text classification, entity extraction, summarization, and sentiment analysis
- **Intelligent RAG Integration**: Automatic detection and processing of domain-specific queries
- **Document Management**: ChromaDB-based vector storage for knowledge retrieval
- **Webhook Notifications**: Asynchronous task status updates
- **Health Monitoring**: Real-time system performance tracking
- **Async Processing**: High-performance asynchronous request handling

## 📋 Prerequisites

Before running this project, ensure you have:

- **Python 3.8+** installed on your system
- **ChromaDB** (automatically installed with the project)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd NLP_Test-A
```

### 2. Create a Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Start the App


### 1. Run the Application

```bash
python main.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Alternative: Using Uvicorn Directly

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 What This Project Does

### Core Functionality

1. **NLP Text Processing**
   - **Text Classification**: Categorize text into different classes
   - **Entity Extraction**: Extract named entities (people, organizations, locations, etc.)
   - **Text Summarization**: Generate concise summaries of longer texts
   - **Sentiment Analysis**: Analyze the emotional tone of text

2. **Intelligent RAG Integration**
   - Automatically detects queries about Deep Learning, Machine Learning, or Agentic AI
   - Uses vector database (ChromaDB) to retrieve relevant documents
   - Enhances responses with contextual information from stored knowledge

3. **Document Management**
   - Add documents to the knowledge base for RAG processing
   - Vector-based similarity search for relevant document retrieval
   - Persistent storage using ChromaDB

### API Endpoints

#### NLP Endpoints (`/nlp/`)
- `POST /nlp/classify` - Classify text into categories
- `POST /nlp/entities` - Extract named entities from text
- `POST /nlp/summarize` - Generate text summaries
- `POST /nlp/sentiment` - Analyze text sentiment

#### RAG Endpoints (`/rag/`)
- `POST /rag/documents/add` - Add documents to knowledge base
- `GET /rag/documents/list` - List stored documents

#### Health & Status
- `GET /` - Welcome message
- `GET /health` - Health check endpoint

### Example Usage

#### Text Sentiment Analysis
```bash
curl -X POST "http://localhost:8000/nlp/sentiment" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "I absolutely love this new product! It works perfectly and exceeded all my expectations.",
       "task_id": "sentiment_analysis_001"
     }'
```

**Expected Response:**
```json
{
  "task_id": "sentiment_analysis_001",
  "status": "completed",
  "result": "The sentiment of this text is overwhelmingly positive. The use of words like 'absolutely love', 'perfectly', and 'exceeded all expectations' indicates strong positive emotions and satisfaction with the product."
}
```

