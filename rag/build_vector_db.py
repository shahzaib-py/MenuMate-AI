import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# ---------- PATHS ----------
DATA_FILE = "data/restaurant_knowledge.txt"
VECTOR_DB_PATH = "rag/vector_store"

# ---------- LOAD DATA ----------
def load_text():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()

# ---------- SPLIT INTO CHUNKS BY SECTION ----------
def split_text(text):
    # Split by "---" separator to keep each section together
    sections = text.split("---")
    # Clean up and filter empty sections
    chunks = [section.strip() for section in sections if section.strip()]
    return chunks

# ---------- BUILD VECTOR DB ----------
def build_vector_db():
    print("Loading restaurant data...")
    text = load_text()

    print("Splitting text into chunks...")
    chunks = split_text(text)

    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    print("Saving vector database...")
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)

    print("[SUCCESS] Vector database created successfully!")

if __name__ == "__main__":
    build_vector_db()
