from google import genai
from google.genai import types as genai_types
from pymilvus import MilvusClient
from config import *
from language import resolve_languages, strip_instruction

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

CRITICAL LANGUAGE RULE: The knowledge base is in English, but you MUST write your entire answer in the
"Response language" given with each question — regardless of the language the question or context is in.
- Gujarati → write ONLY in Gujarati script (ગુજરાતી)
- Hindi → write ONLY in Devanagari script (हिंदी)
- English → write ONLY in English
Keep Article/Amendment numbers as digits, and you may add the English legal term in brackets after
the translated term where it helps (e.g. "મૂળભૂત અધિકારો (Fundamental Rights)").

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


def translate_query_to_english(query: str, source_language: str) -> str:
    """Translate a non-English question to English so it matches the English-only vector store."""
    prompt = (
        f"Translate this {LANGUAGE_NAMES.get(source_language, source_language)} question about the Indian "
        "Constitution into English for a search engine. Ignore any instruction about which language to "
        "answer in. Output only the English question, nothing else.\n\n"
        f"{query}"
    )
    try:
        response = _genai_client.models.generate_content(model=QUERY_TRANSLATION_MODEL, contents=prompt)
        translated = (response.text or "").strip()
        return translated or query
    except Exception as e:
        print(f"Query translation failed, using original query: {e}")
        return query


def generate_answer(query: str, context_chunks: list, input_language: str, output_language: str) -> tuple:
    rag_context = "\n\n---\n\n".join([c["text"] for c in context_chunks]) if context_chunks else "No specific RAG context available."
    in_name = LANGUAGE_NAMES.get(input_language, "English")
    out_name = LANGUAGE_NAMES.get(output_language, "English")

    user_message = f"""RAG Knowledge Base Context (English):
{rag_context}

User Question (written in {in_name}): {query}

Response language: {out_name}

Instructions:
- Use the RAG context above AND your comprehensive knowledge of the Indian Constitution
- Use Google Search to find current developments, recent SC judgments, and latest amendments
- Cite specific Articles, Amendments, and landmark cases
- Write the ENTIRE answer ONLY in {out_name}, even though the question may be in {in_name} and the context is in English
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


def query_rag(user_query: str, input_language: str = None, output_language: str = None) -> dict:
    langs = resolve_languages(user_query, input_language, output_language, SUPPORTED_LANGUAGES)
    in_lang, out_lang = langs["input_language"], langs["output_language"]

    # The vector store is English-only: drop "answer in X" phrasing and translate non-English queries.
    search_query = strip_instruction(user_query, langs["instruction"])
    if in_lang != "en":
        search_query = translate_query_to_english(search_query, in_lang)

    client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    query_embedding = get_embedding(search_query)
    sources = search_documents(client, query_embedding)
    answer, web_sources = generate_answer(user_query, sources, in_lang, out_lang)
    return {
        "response": answer,
        "language": out_lang,
        "input_language": in_lang,
        "output_language": out_lang,
        "output_language_source": langs["output_source"],
        "search_query": search_query,
        "sources": sources,
        "web_sources": web_sources
    }
