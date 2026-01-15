# agents need a goal, state/memory, an llm, and tools to use
# observe -> think -> act ... repeat until stopped
# ReAct pattern is thought: ... -> action: function()

import re
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

TOOLS = {
    "add": add,
    "multiply": multiply,
}

SYSTEM_PROMPT = """
You are a minimal agent.

You must follow this format exactly:

THOUGHT: your reasoning
ACTION: tool_name(arg1=value1, arg2=value2)

If the task is complete, use:
ACTION: finish(result=...)
"""

class Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []

    def build_prompt(self, goal):
        memory_text = "\n".join(str(m) for m in self.memory)

        return f"""
            {SYSTEM_PROMPT}

            Goal: {goal}

            MEMORY:
            {memory_text}
        """
    
    def parse(self, response):
        thought = re.search(r"THOUGHT:(.*)", response).group(1).strip()
        action = re.search(r"ACTION:(.*)", response).group(1).strip()

        name, args_str = action.split("(", 1)
        args_str = args_str.rstrip(")")

        args = {}
        if args_str:
            for pair in args_str.split(","):
                k, v = pair.split("=")
                args[k.strip()] = int(v)
        
        return thought, {"name": name.strip(), "args": args}
    
    def execute(self, action):
        tool = self.tools[action["name"]]
        return tool(**action["args"])
    
    def run(self, goal, max_steps=5):
        for step in range(max_steps):
            prompt = self.build_prompt(goal)
            response = self.llm(prompt)

            thought, action = self.parse(response)

            print(f"\nStep {step + 1}")
            print("Thought:", thought)
            print("Action:", action)

            if action["name"] == "finish":
                return action["args"]["result"]

            result = self.execute(action)
            self.memory.append(result)
        
        raise RuntimeError("Agent did not finish in time")

load_dotenv()
api_key = os.getenv("API_KEY")

model = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=api_key)
agent = Agent(llm=model, tools=TOOLS)
result = agent.run("Add 2 and 3")
print("Final Result:", result)