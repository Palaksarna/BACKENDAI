from fastapi import FastAPI
from .db.chroma_client import ensure_knowledge_base
from .routes.auth import router as auth_router
from .routes.chat import router as chat_router

app = FastAPI()


@app.on_event("startup")
def startup_event():
	ensure_knowledge_base()

app.include_router(auth_router)
app.include_router(chat_router)