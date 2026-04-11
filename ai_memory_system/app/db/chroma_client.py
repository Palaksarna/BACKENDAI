import os
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# PersistentClient guarantees data is written to disk across restarts.
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(name="memory")