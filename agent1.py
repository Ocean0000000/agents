# agents need a goal, state/memory, an llm, and tools to use
# observe -> think -> act ... repeat until stopped
# ReAct pattern is thought: ... -> action: function()

import os
from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter, ValidationError, SecretStr
from typing import Literal
from collections.abc import Callable
from langchain_openai import ChatOpenAI


def add(a: int, b: int) -> int:
    print(f"Adding {a} and {b}")
    return a + b


def multiply(a: int, b: int) -> int:
    print(f"Multiplying {a} and {b}")
    return a * b


TOOLS = {
    "add": add,
    "multiply": multiply,
}


class ToolCall(BaseModel):
    name: Literal["add", "multiply"]
    a: int
    b: int


class Finish(BaseModel):
    result: int


SYSTEM_PROMPT = """
You are a minimal agent. You must use the tools available from the following:

TOOLS = {
    "add": add,
    "multiply": multiply,
}

add(a: int, b: int) => int
multiply(a: int, b: int) => int

I've made a JSON schema for you to give a structured output. You must follow this format exactly.
In your responses, return a single JSON object with no extra text that must match the following schemas depending on your action:

Use this schema to call a tool:
{
    "name": "tool_name",
    "a": value1,
    "b": value2
}

If the task is complete, use this schema:
{
    "result": value
}

If you do not follow the schema exactly, I will tell you to fix your response.
"""


class Agent:
    def __init__(self, llm: ChatOpenAI, tools: dict[str, Callable[[int, int], int]]):
        self.llm = llm
        self.tools = tools
        self.memory = []

    def build_prompt(self, goal: str) -> str:
        memory_text = "\n".join(str(m) for m in self.memory)

        return f"""
            {SYSTEM_PROMPT}

            Goal: {goal}

            MEMORY:
            {memory_text}
        """

    def parse(self, response: str) -> ToolCall | Finish | None:
        adapter = TypeAdapter(ToolCall | Finish)
        structured_output = adapter.validate_json(response)
        if isinstance(structured_output, ToolCall):
            return structured_output
        elif isinstance(structured_output, Finish):
            return structured_output
        return None

    def execute(self, action: ToolCall) -> int:
        return self.tools[action.name](action.a, action.b)

    def run(self, goal: str, max_steps: int = 5) -> int:
        for step in range(max_steps):
            prompt = self.build_prompt(goal)
            response = self.llm.invoke(prompt)
            content = response.content

            if isinstance(content, list):
                pieces: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        pieces.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        pieces.append(part["text"])
                    else:
                        pieces.append(str(part))
                content = "\n".join(pieces)

            try:
                action = self.parse(content)
                print("Prompt:", prompt)
                print(f"\nStep {step + 1}")
                print("Action:", action)

                if isinstance(action, ToolCall):
                    tool_result = self.execute(action)
                    self.memory.append(
                        f"Executed {action.name} with result: {tool_result}"
                    )

                if isinstance(action, Finish):
                    return action.result
            except ValidationError:
                self.memory.append(f"Could not parse response: {content}")
                print(f"Could not parse response: {content}")
                continue

        raise RuntimeError("Agent did not finish in time")


load_dotenv()
api_key = os.getenv("API_KEY")
api_key = SecretStr(api_key) if api_key else None

llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=api_key)
agent = Agent(llm=llm, tools=TOOLS)
result = agent.run("Add 2 and 3")
print("Final Result:", result)
