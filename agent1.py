# agents need a goal, state/memory, an llm, and tools to use
# observe -> think -> act ... repeat until stopped
# ReAct pattern is thought: ... -> action: function()

import re
import os
from dotenv import load_dotenv
from pydantic import SecretStr
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
    
    def parse(self, response: str) -> tuple[str, dict]:
        thought_match = re.search(r"THOUGHT:(.*)", response)
        action_match = re.search(r"ACTION:(.*)", response)

        if not thought_match or not action_match:
            raise ValueError("Response is not in the correct format")

        thought = thought_match.group(1).strip()
        action = action_match.group(1).strip()

        name, args_str = action.split("(", 1)
        args_str = args_str.rstrip(")")

        args = {}
        if args_str:
            for pair in args_str.split(","):
                k, v = pair.split("=")
                args[k.strip()] = int(v)
        
        return thought, {"name": name.strip(), "args": args}
    
    def execute(self, action: dict) -> int:
        tool = self.tools[action["name"]]
        return tool(action["args"]["a"], action["args"]["b"])
    
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
                content = "/n".join(pieces)

            thought, action = self.parse(content)

            print("Prompt:", prompt)
            print(f"\nStep {step + 1}")
            print("Thought:", thought)
            print("Action:", action)

            if action["name"] == "finish":
                return action["args"]["result"]

            result = self.execute(action)
            self.memory.append(f"Result Found: {result}")
        
        raise RuntimeError("Agent did not finish in time")

load_dotenv()
api_key = os.getenv("API_KEY")
api_key = SecretStr(api_key) if api_key else None

model = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=api_key)
agent = Agent(llm=model, tools=TOOLS)
result = agent.run("Add 2 and 3")
print("Final Result:", result)