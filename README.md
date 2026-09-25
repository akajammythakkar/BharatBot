# BharatBot
> Multilingual RAG Chatbot — English | हिंदी | ગુજરાતી

## Stack
- **LLM**: Gemini 2.5 Pro
- **Embeddings**: Google text-embedding-004
- **Vector DB**: Zilliz Cloud (Milvus)
- **Backend**: FastAPI
- **Frontend**: Pure HTML/CSS/JS (Glassmorphism UI)
- **Deploy**: Google Cloud Run

## Cross-lingual answers

The knowledge base is English-only, but you can ask in one language and get the answer in another.
Two optional dropdowns above the chat box ("Ask in" and "Answer in") control this; both default to Auto.

- **Answer language**, first match wins: an instruction in the question ("Explain Article 21 in Gujarati",
  "ગુજરાતીમાં જવાબ આપો", "hindi mein batao") → the "Answer in" dropdown → the language of the question → English.
- **Question language**: the "Ask in" dropdown → detected from the script (Gujarati / Devanagari) → English.
- Non-English questions are translated to English (`QUERY_TRANSLATION_MODEL`) before the vector search.

API: `POST /chat` with `{"message": "...", "input_language": "auto|en|hi|gu", "output_language": "auto|en|hi|gu"}`.
The old `language` field is still accepted as the output language.

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
