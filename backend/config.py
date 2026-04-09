import os
from dotenv import load_dotenv
load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "bharatbot-admin-2024")

COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072  # gemini-embedding-001 output dimension
GENERATION_MODEL = "gemini-2.5-pro"  # primary; rag.py falls back to gemini-2.5-flash
TOP_K = 5
BATCH_SIZE = 50

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati"
}
