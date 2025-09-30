from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import Any, Optional
import httpx

query_api = "http://localhost:8000/query"
# Use synchronous client
client = httpx.Client(timeout=httpx.Timeout(30.0))

def call_query_api(payload: dict[str, Any]) -> Any:
    """call query api and send request payload for successful query"""
    try:
        response = client.post(query_api, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        return result
    except httpx.HTTPStatusError as e:
        print("RAG API returned an error status")
        raise
    except httpx.RequestError as e:
        print("Network error while calling Query API")
        raise
    except ValueError as e:  # JSON decode error
        print("Invalid JSON in Query API response")
        raise

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def query_tool(query: str) -> Any:
    """Use this tool when the user asks questions related to 'lomma', 'lomaa', 'lomaa it', 'lomaa it solutions', or any general information queries. This tool searches the RAG system to provide relevant answers based on the stored knowledge base."""
    payload = {"query": query}
    res = call_query_api(payload)
    return res['answer']

# Tool creation
tools = [multiply, add, subtract, query_tool]