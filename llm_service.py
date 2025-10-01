import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Any, Dict
import httpx

from langchain.chat_models import init_chat_model
from tools import multiply, add, subtract, query_tool

# Load env vars
load_dotenv(override=True)

# Ensure Google key
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

# Persistent HTTP client
client = httpx.Client(timeout=httpx.Timeout(30.0))


class ToolRunner:
    def __init__(self):
        # Initialize model once
        self.model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

        # Bind tools
        self.tools = [multiply, add, subtract, query_tool]
        self.model_with_tools = self.model.bind_tools(self.tools)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Execute model + tool logic for given user input, supporting multi-step tool chaining"""
        history = [user_input]
        final_result = None
        tools_used = []

        while True:
            response = self.model_with_tools.invoke("\n".join(history))
            print("Raw tool_calls:", response.tool_calls)
            print("Raw LLM content:", response.content)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print(f"Invoking tool: {tool_call['name']} with args: {tool_call['args']}")
                    tools_used.append(tool_call["name"])
                    if tool_call["name"] == "multiply":
                        result = multiply.invoke(tool_call["args"])
                    elif tool_call["name"] == "add":
                        result = add.invoke(tool_call["args"])
                    elif tool_call["name"] == "subtract":
                        result = subtract.invoke(tool_call["args"])
                    elif tool_call["name"] == "query_tool":
                        result = query_tool.invoke(tool_call["args"])
                    # Add result to history for next round
                    history.append(f"The result of {tool_call['name']} is {result}.")
                    final_result = result
            else:
                print("LLM Answer:", response.content)
                if response.content:
                    final_result = response.content
                break

        return {
            "type": "tool_call",
            "tools_used": tools_used,
            "answer": final_result
        }

tool_runner = ToolRunner()  # initialized once
