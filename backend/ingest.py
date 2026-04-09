import pandas as pd
from google import genai
from google.genai import types as genai_types
from pymilvus import MilvusClient, DataType
from langdetect import detect, DetectorFactory
from config import *
import time

DetectorFactory.seed = 42
_genai_client = genai.Client(api_key=GEMINI_API_KEY)


def get_milvus_client():
    # Try token auth first, fall back to user/password
    try:
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        client.list_collections()  # test connection
        return client
    except Exception:
        user, password = ZILLIZ_TOKEN.split(":", 1)
        return MilvusClient(uri=ZILLIZ_URI, user=user, password=password)


def detect_language(text: str) -> str:
    try:
        lang = detect(str(text))
        return lang if lang in ["en", "hi", "gu"] else "en"
    except Exception:
        return "en"


def embed_texts(texts: list) -> list:
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type="retrieval_document")
    )
    return [e.values for e in result.embeddings]


def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("language", DataType.VARCHAR, max_length=20)
    schema.add_field("source_row", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def load_dataset(filepath: str) -> list:
    df = pd.read_excel(filepath, engine="openpyxl")
    print("Columns found:", df.columns.tolist())
    print(df.head(3))

    text_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["question", "answer", "text", "content", "prompt", "response", "input", "output"]
    )]
    if not text_cols:
        text_cols = df.columns.tolist()
    print(f"Using columns for RAG: {text_cols}")

    docs = []
    for i, row in df.iterrows():
        parts = []
        for col in text_cols:
            val = str(row[col]).strip()
            if val and val.lower() not in ["nan", "none", ""]:
                parts.append(f"{col}: {val}")
        doc_text = "\n".join(parts)
        if doc_text.strip():
            docs.append({
                "text": doc_text[:65000],
                "language": detect_language(doc_text),
                "source_row": int(i)
            })
    print(f"Total documents prepared: {len(docs)}")
    return docs


def ingest(filepath: str = "dataset_gemini_3.1_flash_balanced.xlsx"):
    client = get_milvus_client()
    setup_collection(client)

    docs = load_dataset(filepath)
    total = len(docs)

    for start in range(0, total, BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        texts = [d["text"] for d in batch]

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            print(f"Embedding error at batch {start}: {e}")
            time.sleep(5)
            embeddings = embed_texts(texts)

        data = [
            {
                "text": batch[j]["text"],
                "embedding": embeddings[j],
                "language": batch[j]["language"],
                "source_row": batch[j]["source_row"]
            }
            for j in range(len(batch))
        ]
        client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"Inserted rows {start} to {start + len(batch)} / {total}")
        time.sleep(0.5)

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
