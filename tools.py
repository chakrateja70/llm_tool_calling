import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

# Configure API key
api_key = "your api key here"

# Wrap Gemini in a LangChain-compatible class
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Define a tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# Bind the tool to the model
llm_with_tools = model.bind_tools([multiply])

# Ask a question
result = llm_with_tools.invoke("What is 2 multiplied by 3?")

# Check if tool was called and extract the answer
if result.tool_calls:
    for call in result.tool_calls:
        if call["name"] == "multiply":
            a = call["args"]["a"]
            b = call["args"]["b"]
            answer = a * b
            print(f"Answer: {answer}")
else:
    print(f"Answer: {result.content}")
