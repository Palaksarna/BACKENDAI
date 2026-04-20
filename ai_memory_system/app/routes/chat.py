from fastapi import APIRouter
from ..models.schema import ChatRequest, ChatResponse
from ..services.memory_service import (
    retrieve_relevant_documents,
)
from ..services.llm_service import generate_response
from ..services.fact_memory import (
    build_prompt,
    extract_facts,
    get_promoted_facts,
    update_fact_store,
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message
    facts = extract_facts(message)
    if facts:
        update_fact_store(facts)

    promoted_facts = get_promoted_facts()
    retrieved_context = retrieve_relevant_documents(message)
    prompt = build_prompt(
        message=message,
        retrieved_context=retrieved_context,
        promoted_facts=promoted_facts,
    )

    response = generate_response(prompt)

    return {
        "response": response,
        "retrieved_context": retrieved_context,
    }