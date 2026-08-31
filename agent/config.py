from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT/"model/checkpoint-31250"

OPENAI_MODEL = "gpt-4.1-mini"

RAG_DOCUMENT_PATH = PROJECT_ROOT/"rag/documents/student.txt"
RAG_VECTORSTORE_PATH = PROJECT_ROOT/"rag/vectorstore"

SOURCE_LANGUAGE_MBART = "en_XX"
TARGET_LANGUAGE_MBART = "uk_UA"
