# doing what i did in agent1.py but using langchain

import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

load_dotenv()
api_key_str = os.getenv("API_KEY")
api_key = SecretStr(api_key_str) if api_key_str else None

loader = DirectoryLoader(
    "./rag_dataset",
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
)

docs = loader.load()
print(f"Loaded {len(docs)} documents.")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

split_docs: list[Document] = []

for doc in docs:
    split_docs.extend(splitter.split_text(doc.page_content))
docs = split_docs

print(f"Split into {len(docs)} chunks.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)


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
def retrieve(query: str) -> str:
    """
    Retrieves relevant information from the vector store based on the query.
    """
    print(f"Retrieving information for query: {query}")

    retriever = vector_store.as_retriever(kwargs={"search_kwargs": {"k": 1}})

    results = retriever.invoke(query)

    if results:
        return "\n\n".join([doc.page_content for doc in results])

    return "No relevant information found."


class AIResponse(BaseModel):
    answer: str


TOOLS = {
    "think": think,
    "add": add,
    "multiply": multiply,
    "retrieve": retrieve,
}

SYSTEM_PROMPT = """
You are a minimal agent with tools. Always use the thought tool before you do anything else.
"""


async def main():
    client = MultiServerMCPClient(
        {
            "sis": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    mcp_tools = await client.get_tools()
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    agent = create_agent(
        model=model,
        response_format=AIResponse,
        system_prompt=SYSTEM_PROMPT,
        tools=[TOOLS["think"], TOOLS["add"], TOOLS["multiply"], TOOLS["retrieve"], *mcp_tools],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Add 2 and 3")]})
    if result["structured_response"]:
        print(result["structured_response"])

    rag_result = await agent.ainvoke(
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
    
    mcp_result = await agent.ainvoke({"messages": [HumanMessage("Add a student named Ocean Van"), HumanMessage("What is the GPA of Ocean Van?")]})
    if mcp_result["structured_response"]:
        print(mcp_result["structured_response"])

if __name__ == "__main__":
    asyncio.run(main())