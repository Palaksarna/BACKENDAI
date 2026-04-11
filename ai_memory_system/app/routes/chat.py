from fastapi import APIRouter
from ..models.schema import ChatRequest
from ..services.memory_service import store_memory, retrieve_memory, forget_memory
from ..services.llm_service import generate_response

router = APIRouter()

@router.post("/chat")
def chat(req: ChatRequest):
    user_id = req.user_id
    message = req.message
    stored_items = []

    # Store every non-empty user message so retrieval has reliable context.
    stored = store_memory(user_id, message)
    if stored:
        stored_items.append(stored)

    # retrieve memory
    memories = retrieve_memory(user_id, message)

    memory_text = "\n".join([m[0] for m in memories])

    # construct prompt
    prompt = f"""
    User profile:
    {memory_text}

    User question:
    {message}
    """

    response = generate_response(prompt)

    # forgetting step
    forget_memory()

    return {
        "response": response,
        "stored_memory": stored_items,
        "retrieved_memory": [m[0] for m in memories]
    }