# BharatBot
> Multilingual RAG Chatbot — English | हिंदी | ગુજરાતી

## Stack
- **LLM**: Gemini 2.5 Pro
- **Embeddings**: Google text-embedding-004
- **Vector DB**: Zilliz Cloud (Milvus)
- **Backend**: FastAPI
- **Frontend**: Pure HTML/CSS/JS (Glassmorphism UI)
- **Deploy**: Google Cloud Run

## Setup

```bash
conda create -n rag-chatbot python=3.11 -y
conda activate rag-chatbot
pip install -r requirements.txt
cp .env.example .env  # fill in credentials
```

## Ingest Data

```bash
cd backend
python ingest.py
```

## Run Locally

```bash
uvicorn backend.main:app --reload --port 8080
```

## Deploy

```bash
gcloud builds submit --config cloudbuild.yaml
```
