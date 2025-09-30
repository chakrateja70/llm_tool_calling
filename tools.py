import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

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

# Tool creation
tools = [multiply, add, subtract]

# Tool binding
model_with_tools = model.bind_tools(tools)  

# Tool calling
user_input = "what is LLM?"
response = model_with_tools.invoke(user_input)

if response.tool_calls:
    # execute tools
    final_result = None
    for tool_call in response.tool_calls:
        if tool_call["name"] == "multiply":
            final_result = multiply.invoke(tool_call["args"])
        elif tool_call["name"] == "add":
            final_result = add.invoke(tool_call["args"])
        elif tool_call["name"] == "subtract":
            final_result = subtract.invoke(tool_call["args"])
    print("Final Answer:", final_result)
else:
    # normal LLM response
    print("LLM Answer:", response.content)
