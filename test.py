from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. Define a simple tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
api_key = "AIzaSyDZrvu-I9yQH-JRQI6c-GA9jKUtfIqNhjM"

# 2. Create Gemini model wrapped for LangChain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 3. Bind the tool
llm_with_tools = llm.bind_tools([multiply])

# 4. Invoke with a plain user input (Gemini will call the tool if needed)
result = llm_with_tools.invoke("What is 2 multiplied by 3?")
print("🔹 Model Output:", result)
