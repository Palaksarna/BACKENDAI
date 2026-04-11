import json

from fastapi import APIRouter
from ..models.schema import ChatRequest
from ..services.memory_service import (
    forget_memory,
    get_neural_context,
    process_memory_buffer,
    retrieve_memory,
    store_memory,
)
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
    promoted_memory_ids = process_memory_buffer()
    neural_context = get_neural_context(message)

    memory_text = "\n".join([m[0] for m in memories])
    neural_text = json.dumps(neural_context)

    # construct prompt
    prompt = f"""
    User profile:
    {memory_text}

    Neural layer context:
    {neural_text}

    User question:
    {message}
    """

    response = generate_response(prompt)

    # forgetting step
    forget_memory()

    return {
        "response": response,
        "stored_memory": stored_items,
        "retrieved_memory": [m[0] for m in memories],
        "promoted_memory_ids": promoted_memory_ids,
        "neural_context": neural_context,
    }