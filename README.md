# Agentic-RAG: Wikipedia Chatbot with Multi-Agent Reasoning

A sophisticated Retrieval-Augmented Generation (RAG) system that leverages multiple AI agents to answer questions about Wikipedia articles with enhanced accuracy and reasoning.

## 🎯 Overview

This project implements an agentic RAG pipeline using LangChain and LangGraph, where multiple specialized agents work together to process user queries and retrieve relevant information from Wikipedia:

1. **Agent 1 — Query Rewrite**: Optimizes user queries for better Wikipedia retrieval
2. **Agent 2 — Knowledge Update**: Identifies and retrieves additional Wikipedia articles as needed
3. **Agent 3 — Multiple Retrieval**: Determines optimal document retrieval parameters
4. **Generator**: Produces final answers based on retrieved documents

## 🏗️ Architecture

### Workflow
![Agentic RAG Workflow](images/Agentic%20RAG.png)

### Backend Components

- **agent.py**: Core agent definitions and orchestration logic
- **backend.py**: FastAPI server exposing agent endpoints
- **app.py**: Streamlit frontend for interactive chatbot

### RAG Components

- **rag/wikipedia_processor.py**: Processes and indexes Wikipedia articles
- **rag/chunker.py**: Hierarchical document chunking
- **rag/retriever.py**: Hybrid retrieval logic

### Storage

- **storage/vector_store.py**: FAISS-based vector storage
- **storage/doc_store.py**: SQLite document metadata storage

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Generative AI API key
- FastAPI and Streamlit

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YeeJingGan/Agentic-RAG.git
cd Agentic-RAG
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_api_key_here
LLM_MODEL_ID=your_llm_model_name
EMBEDDING_MODEL_ID=your_embedding_model_name
```

### Running the Application

1. Start the FastAPI backend:
```bash
python backend.py
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## 💡 How It Works

### Query Processing Pipeline

```
User Query
    ↓
[Agent 1] Query Rewrite (optimizes the question)
    ↓
[Agent 2] Knowledge Base Update (adding more Wikipedia articles)
    ↓
[Agent 3] Reasoning (determines retrieval parameters)
    ↓
[Generator] Final Answer Generation
    ↓
Chat Response with Citations
```


## 📁 Project Structure

```
.
├── agent.py                 # Main agent definitions
├── backend.py              # FastAPI server
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── rag/
│   ├── chunker.py         # Document chunking logic
│   ├── retriever.py       # Retrieval system
│   └── wikipedia_processor.py
├── storage/
│   ├── vector_store.py    # FAISS vector database
│   └── doc_store.py       # SQLite document store
└── faiss_index/           # Vector index storage
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent1` | POST | Query rewriting agent |
| `/agent2` | GET | Knowledge base update |
| `/agent3` | GET | Document retrieval reasoning |
| `/state` | GET | Get current agent state |
| `/query` | GET | Generate final answer (streaming) |

## 📦 Dependencies

Key libraries:
- **LangChain**: Agent and RAG framework
- **LangGraph**: Graph-based orchestration
- **FastAPI**: Backend server
- **Streamlit**: Frontend UI
- **FAISS**: Vector similarity search
- **Google Generative AI**: LLM provider

## 🛠️ Configuration

Edit `.env` to customize:

```env
GOOGLE_API_KEY=your_key_here
REQUEST_URL=http://127.0.0.1:8000
LLM_MODEL_ID=gemini-2.5-flash-lite
```

## 📄 Reading Materials

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401) 
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/pdf/2312.10997)
- [Advanced RAG: Small-to-Big Retrieval](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4)
