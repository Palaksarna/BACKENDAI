from fastapi import APIRouter, Query
from ..models.schema import ChatRequest, ChatResponse
from ..services.memory_service import (
    retrieve_relevant_documents,
)
from ..services.llm_service import generate_response
from ..services.fact_memory import (
    build_prompt,
    extract_facts,
    get_fact_debug_snapshot,
    update_fact_store,
)
from ..ml.neural_fact_memory import retrieve_neural_facts

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message
    
    # Extract and store facts for neural training
    # (System will learn importance from usage, not explicit profile)
    facts = extract_facts(message)
    if facts:
        update_fact_store(facts)

    # Retrieve context via neural models (learned importance)
    neural_facts = retrieve_neural_facts(message, top_k=5)
    
    # Retrieve context via RAG (similarity-based)
    retrieved_context = retrieve_relevant_documents(message)
    
    # Build prompt with neural-learned facts, not hardcoded profiles
    prompt = build_prompt(
        message=message,
        retrieved_context=retrieved_context,
        promoted_facts=[],  # Not used - neural model decides importance
        neural_facts=neural_facts,
    )

    response = generate_response(prompt)

    return {
        "response": response,
        "retrieved_context": retrieved_context,
    }


@router.get("/memory/facts/debug")
def memory_facts_debug(
    query: str = Query(default="", description="Optional query for neural fact preview"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of fact entries to return"),
):
    return get_fact_debug_snapshot(query=query, limit=limit)