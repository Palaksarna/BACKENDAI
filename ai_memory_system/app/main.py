from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.chroma_client import ensure_knowledge_base
from .routes import auth, chat


app = FastAPI(title="AI Memory Backend")


@app.on_event("startup")
def startup_event() -> None:
	ensure_knowledge_base()


app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


app.include_router(auth.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict[str, str]:
	return {"status": "ok", "message": "Backend is running"}