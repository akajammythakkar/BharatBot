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

SYSTEM_PROMPT = """You are BharatBot (भारतबॉट), an expert AI assistant specializing in the Constitution of India.

YOUR EXPERTISE:
- All 448 Articles of the Indian Constitution
- 12 Schedules, 25 Parts, 104 Amendments
- Fundamental Rights (Part III), Fundamental Duties (Part IVA)
- Directive Principles of State Policy (Part IV)
- Constitutional history, Constituent Assembly Debates
- Landmark Supreme Court judgments (Kesavananda Bharati, Maneka Gandhi, etc.)
- Constitutional amendments and their impact
- Comparative constitutional law where relevant to India

HOW TO ANSWER:
1. Use the provided RAG context from the knowledge base
2. Draw on your comprehensive knowledge of Indian Constitutional law
3. Use real-time web search results for current developments and recent judgments
4. Always cite relevant Articles (e.g., "Article 21", "Article 32") and Amendments
5. Mention landmark cases when relevant
6. Be accurate, thorough, and educational

CRITICAL LANGUAGE RULE: You MUST respond ONLY in the language the user wrote in:
- User writes in Gujarati (ગુજરાતી) → respond ONLY in Gujarati
- User writes in Hindi (हिंदी) → respond ONLY in Hindi
- User writes in English → respond ONLY in English

FORMAT: Structure complex answers with clear headings. Use bullet points for lists of rights/articles. Be comprehensive yet accessible."""


def get_embedding(text: str) -> list:
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="retrieval_query")
    )
    return result.embeddings[0].values


def search_documents(client: MilvusClient, query_embedding: list) -> list:
    try:
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
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def generate_answer(query: str, context_chunks: list, language: str) -> tuple:
    rag_context = "\n\n---\n\n".join([c["text"] for c in context_chunks]) if context_chunks else "No specific RAG context available."
    lang_name = LANGUAGE_NAMES.get(language, "English")

    user_message = f"""RAG Knowledge Base Context:
{rag_context}

User Question (in {lang_name}): {query}

Instructions:
- Use the RAG context above AND your comprehensive knowledge of the Indian Constitution
- Use Google Search to find current developments, recent SC judgments, and latest amendments
- Cite specific Articles, Amendments, and landmark cases
- Respond ONLY in {lang_name}
- Be thorough and educational"""

    models_to_try = [GENERATION_MODEL, "gemini-2.5-pro", "gemini-2.5-flash"]
    last_error = None

    for model_name in models_to_try:
        # First try with Google Search grounding
        try:
            config = genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            )
            response = _genai_client.models.generate_content(
                model=model_name,
                contents=user_message,
                config=config
            )

            web_sources = []
            if response.candidates:
                gm = response.candidates[0].grounding_metadata
                if gm and hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            web_sources.append({
                                "title": getattr(chunk.web, "title", "") or "",
                                "uri": getattr(chunk.web, "uri", "") or ""
                            })

            return response.text, web_sources

        except Exception as e:
            last_error = e
            print(f"Model {model_name} with grounding failed: {e}")

            # Try same model without grounding
            try:
                config = genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                )
                response = _genai_client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=config
                )
                return response.text, []
            except Exception as e2:
                last_error = e2
                print(f"Model {model_name} without grounding also failed: {e2}")
                continue

    raise RuntimeError(f"All generation models failed. Last error: {last_error}")


def query_rag(user_query: str, language: str = "en") -> dict:
    client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    query_embedding = get_embedding(user_query)
    sources = search_documents(client, query_embedding)
    answer, web_sources = generate_answer(user_query, sources, language)
    return {
        "response": answer,
        "language": language,
        "sources": sources,
        "web_sources": web_sources
    }
