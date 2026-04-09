from google import genai
from google.genai import types as genai_types
from pymilvus import MilvusClient
from config import *

_genai_client = genai.Client(api_key=GEMINI_API_KEY)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "gu": "Gujarati (ગુજરાતી)"
}

SYSTEM_PROMPT = """You are BharatBot, a helpful multilingual AI assistant.
You have been given context retrieved from a knowledge base.
CRITICAL RULE: You MUST respond in the EXACT SAME language the user wrote their question in.
- If the user wrote in Gujarati → respond ONLY in Gujarati
- If the user wrote in Hindi → respond ONLY in Hindi
- If the user wrote in English → respond ONLY in English
Use the provided context to answer accurately. If the context doesn't contain the answer, say so honestly in the user's language.
Be concise, helpful, and friendly."""


def get_embedding(text: str) -> list:
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="retrieval_query")
    )
    return result.embeddings[0].values


def search_documents(client: MilvusClient, query_embedding: list) -> list:
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=TOP_K,
        output_fields=["text", "language", "source_row"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}}
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit["entity"]["text"],
            "language": hit["entity"]["language"],
            "source_row": hit["entity"]["source_row"],
            "score": round(hit["distance"], 4)
        })
    return hits


def generate_answer(query: str, context_chunks: list, language: str) -> str:
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
    lang_name = LANGUAGE_NAMES.get(language, "English")

    user_message = f"""Context from knowledge base:
{context}

User question (in {lang_name}): {query}

Remember: Respond ONLY in {lang_name}."""

    models_to_try = [
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.0-pro-exp",
        "gemini-1.5-pro",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = _genai_client.models.generate_content(
                model=model_name,
                contents=user_message,
                config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"Model {model_name} failed: {e}. Trying next...")
            continue

    raise RuntimeError(f"All generation models failed. Last error: {last_error}")


def query_rag(user_query: str, language: str = "en") -> dict:
    client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    query_embedding = get_embedding(user_query)
    sources = search_documents(client, query_embedding)
    answer = generate_answer(user_query, sources, language)
    return {
        "response": answer,
        "language": language,
        "sources": sources
    }
