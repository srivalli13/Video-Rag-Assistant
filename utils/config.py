import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Models ---
EMBEDDING_MODEL = "BAAI/bge-m3"
WHISPER_MODEL = "base"
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Retrieval hyperparameters ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5
