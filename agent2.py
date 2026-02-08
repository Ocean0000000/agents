# doing what i did in agent1.py but using langchain

import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_chroma import Chroma


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


@tool
def rag(query: str) -> str:
    """
    Retrieves information based on a query.
    """
    print(f"Retrieving information for query: {query}")
    return f"Information retrieved for query: {query}"


class AIResponse(BaseModel):
    answer: str


TOOLS = {
    "think": think,
    "add": add,
    "multiply": multiply,
    "rag": rag,
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
    tools=[TOOLS["think"], TOOLS["add"], TOOLS["multiply"], TOOLS["rag"]],
)
result = agent.invoke({"messages": [HumanMessage("Add 2 and 3")]})
if result["structured_response"]:
    print(result["structured_response"])

loader = DirectoryLoader(
    "./rag_dataset",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    loader_kwargs={"mode": "elements", "strategy": "fast"},
    show_progress=True,
    use_multithreading=True,
)

docs = loader.load()
print(f"Loaded {len(docs)} documents.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

rag_result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "What is the recommended brew time for a double shot on the Atlas machine?"
            )
        ]
    }
)
if rag_result["structured_response"]:
    print(rag_result["structured_response"])
