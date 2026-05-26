# DocuMind-RAG

DocuMind-RAG is a Local RAG (Retrieval-Augmented Generation) backend system that allows users to upload PDF documents and ask AI-powered questions from them using local LLMs through Ollama.

The project uses semantic search and vector embeddings to retrieve relevant document context before generating responses.

---

## Features

- PDF Upload Support
- AI-Powered Question Answering
- Local LLM Integration using Ollama
- Semantic Search
- Vector Embeddings
- FAISS Vector Database
- Context-Aware Responses
- Lightweight FastAPI Backend

---

## Tech Stack

- FastAPI
- FAISS
- sentence-transformers
- Ollama
- Llama3 / Mistral
- PyMuPDF
- Python

---

## How It Works

1. Upload PDF documents
2. Extract and split document text into chunks
3. Generate embeddings using sentence-transformers
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks using semantic similarity
6. Send retrieved context to Ollama-powered LLM
7. Generate AI response

---

## APIs

### Upload PDF

```http
POST /upload-pdf
```

### Ask Question

```http
POST /ask-question
```

### Health Check

```http
GET /health
```

---

## Run Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama run llama3
```

### Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

---

## Swagger API Docs

```text
http://127.0.0.1:8000/docs
```

---

## Future Improvements

- Conversation Memory
- Hybrid Search
- Multi-PDF Support
- Streaming Responses
- Authentication System
- Cloud Deployment

---

## Project Goal

This project focuses on learning and implementing modern AI engineering concepts such as:

- RAG Architecture
- Embeddings
- Semantic Search
- Vector Databases
- Local LLM Inference
- AI Backend Development
