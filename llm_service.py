import os
from groq import Groq, APIError
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment variables.")

client = Groq(api_key=api_key)

print("Client initialized successfully.")
formatted_prompt = "what is difference between RAG and Fine Tuning?"
try:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=300,
    )
    print("LLM is processing...")
    answer = chat_completion.choices[0].message.content.strip()
    print("Answer:", answer)
except APIError as e:
    print(f"APIError occurred: {e}")