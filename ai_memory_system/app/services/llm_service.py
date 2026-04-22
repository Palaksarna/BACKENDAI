import os
from dotenv import load_dotenv

# Load local .env values before resolving API credentials.
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY", "")
model_name = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

def generate_response(prompt):
    """
    Generate response using HuggingFace Inference API or mock response.
    Falls back to mock response if API key is unavailable.
    """
    if not api_key:
        # Mock response for testing without HF credentials
        return "Based on the information provided, I can assist you with that. Your facts have been remembered and will be used for future context."
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            top_p=0.9,
            max_tokens=200
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback: return mock response
        print(f"LLM service error (model={model_name}): {e}")
        return "I've processed your message and updated my memory with the information you provided. How can I help you further?"