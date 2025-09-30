import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import Any, Optional
import httpx

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Use a stable model that supports tool calling
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

login_api = "http://localhost:8000/login"
query_api = "http://localhost:8000/query"
# Use synchronous client
client = httpx.Client(timeout=httpx.Timeout(30.0))

def call_login_api(payload: dict[str, Any]) -> Any:
    """call login api and send request payload for successful login"""
    try:
        response = client.post(login_api, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        return result
    except httpx.HTTPStatusError as e:
        print("RAG API returned an error status")
        raise
    except httpx.RequestError as e:
        print("Network error while calling Login API")
        raise
    except ValueError as e:  # JSON decode error
        print("Invalid JSON in Login API response")
        raise
    
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

# @tool
# def answer_question(question: str) -> str:
#     """Answer a question using the LLM."""
#     response = model.invoke(question)
#     return response.content

@tool
def login_tool(name: Optional[str] = None, age: Optional[int] = None) -> str:
    """
    Use this tool when the user asks about registration, login, sign in, sign off, or authentication related to Lomaa IT Solutions 
    (including variations like 'Lomaa', 'lomaa it solutions', or simply 'lomaa').  

    This tool requires two input parameters:
    - name: user's name (string)
    - age: user's age (integer)

    If the user mentions any of these actions but does not provide both name and age, ask them to supply the missing information.
    """
    # Check if required parameters are provided and valid
    if not name or not age or name == "" or age == 0:
        return "To login to Lomaa IT Solutions, I need both your name and age. Please provide: 1) Your name (as a string) 2) Your age (as a number)"
    
    payload = {"name": name, "age": age}
    print(f"Calling login API with payload: {payload}")
    try:
        response = call_login_api(payload)
        return response['message']
    except Exception as e:
        return f"Login failed: {str(e)}"
    
@tool
def query_tool(query: str) -> Any:
    """Use this tool when the user asks questions related to 'lomma', 'lomaa', 'lomaa it', 'lomaa it solutions', or any general information queries. This tool searches the RAG system to provide relevant answers based on the stored knowledge base."""
    payload = {"query": query}
    res = call_query_api(payload)
    return res['answer']

# Tool creation
tools = [multiply, add, subtract, login_tool, query_tool]

# Tool binding
model_with_tools = model.bind_tools(tools)  

# Tool calling
user_input = "login with teja and his age is 21"
response = model_with_tools.invoke(user_input)

if response.tool_calls:
    # execute tools
    print("Tool calls detected. Executing tools...")
    
    final_result = None
    for tool_call in response.tool_calls:
        print(f"Invoking tool: {tool_call['name']} with args: {tool_call['args']}")
        if tool_call["name"] == "multiply":
            final_result = multiply.invoke(tool_call["args"])
        elif tool_call["name"] == "add":
            final_result = add.invoke(tool_call["args"])
        elif tool_call["name"] == "subtract":
            final_result = subtract.invoke(tool_call["args"])
        # elif tool_call["name"] == "answer_question":
        #     final_result = answer_question.invoke(tool_call["args"])
        elif tool_call["name"] == "login_tool":
            final_result = login_tool.invoke(tool_call["args"])
        elif tool_call["name"] == "query_tool":
            final_result = query_tool.invoke(tool_call["args"])

    print("Final Answer:", final_result)
else:
    # normal LLM response
    print("LLM Answer:", response.content)
