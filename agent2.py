# doing what i did in agent1.py but using langchain

import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage


@tool
def think(thought: str) -> str:
    """
    Processes a thought and returns a response
    """
    print(f"Thinking about: {thought}")
    return f"I thought about: {thought}"


@tool
def add(a: int, b: int) -> int:
    """
    Adds two integers.
    """
    print(f"Adding {a} and {b}")
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    """
    print(f"Multiplying {a} and {b}")
    return a * b


class AIResponse(BaseModel):
    answer: str


TOOLS = {
    "think": think,
    "add": add,
    "multiply": multiply,
}

SYSTEM_PROMPT = """
You are a minimal agent with tools. Always use the thought tool before you do anything else.
"""

load_dotenv()
api_key = os.getenv("API_KEY")
api_key = SecretStr(api_key) if api_key else None

model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
agent = create_agent(
    model=model,
    response_format=AIResponse,
    system_prompt=SYSTEM_PROMPT,
    tools=[TOOLS["think"], TOOLS["add"], TOOLS["multiply"]],
)
result = agent.invoke({"messages": [HumanMessage("Add 2 and 3")]})
if result["structured_response"]:
    print(result["structured_response"])
