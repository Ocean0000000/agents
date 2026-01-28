# doing what i did in agent1.py but using langchain

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage

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

TOOLS = {
    "add": add,
    "multiply": multiply,
}

SYSTEM_PROMPT = """
You are a minimal agent. You must use the tools available from the following:

TOOLS = {
    "add": add,
    "multiply": multiply,
}

add(a: int, b: int) => int
multiply(a: int, b: int) => int

You must follow this format exactly:

THOUGHT: your reasoning
ACTION: tool_name(a=value1, b=value2)

If you see Result Found: ... in your memory, you can use that to help you think.

If the task is complete, use:
ACTION: finish(result=...)
"""

load_dotenv()
api_key = os.getenv("API_KEY")
api_key = SecretStr(api_key) if api_key else None

model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[TOOLS["add"], TOOLS["multiply"]],
)
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Add 2 and 3"
        }
    ]
})
messages = result["messages"]
print("Final Result:", result)

last_ai = next(m for m in reversed(messages) if isinstance(m, AIMessage))
text = last_ai.content  # e.g. "ACTION: finish(result=5)"
if "finish(result=" in text and type(text) is str:
    answer = text.split("finish(result=")[-1].rstrip(")")
    print(answer)