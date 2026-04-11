import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load local .env values before resolving API credentials.
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")

client = InferenceClient(api_key=api_key)

def generate_response(prompt):
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1:novita",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=200
    )

    return completion.choices[0].message.content