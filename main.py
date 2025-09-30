from fastapi import FastAPI
from typing import Any, Dict
from llm_service import tool_runner

app = FastAPI()

@app.post("/tool")
def tool_endpoint(payload: Dict[str, Any]):
    """Endpoint to handle tool calling"""
    user_input = payload.get("query", "")
    if not user_input:
        return {"error": "Missing 'query' in request payload"}

    result = tool_runner.run(user_input)
    return result