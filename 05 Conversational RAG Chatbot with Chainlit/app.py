import os
from dotenv import load_dotenv
load_dotenv()

# Route OpenAI-compatible clients (used by RAGWire's metadata LLM) to OpenRouter
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from ragwire import RAGWire
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# import chainlit and supporting packages
import chainlit as cl
from typing import Optional
import tempfile, os

rag = RAGWire("5config_openrouter_qdrant.yaml")

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields, stored values, and filter suggestions for a query.

    Call this before search_documents when the query involves a specific company,
    year, or document type. Skip for purely semantic queries.
    """
    return rag.get_filter_context(query)

@tool
def search_documents(query: str, filters=None):
    """Search the document knowledge base for relevant information.

    Args:
        query: The search query
        filters: Optional metadata filters as a dict (e.g. {"company_name": "apple inc.", "fiscal_year": 2024}).
    """
    import json

    if isinstance(filters, str):
        try:
            filters = json.loads(filters)
        except json.JSONDecodeError:
            filters = None

    results = rag.retrieve(query=query, filters=filters)

    if not results:
        return "No relevant information is found!"
    else:
        return results 

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
memory = InMemorySaver()

SYSTEM_PROMPT = """
    You are a helpful document assistant. 
    For complex questions, break them down into simpler sub-questions and answer each one before forming a final answer.
    Always call search_documents to find information before answering.
    If the query mentions a company, year, or document type, call get_filter_context first.
    If no documents are found, say so honestly — never make up an answer. 
    Always mention the source document in your answer."""

# this will execute on the start of app/ui
@cl.on_chat_start
async def on_chat_start():
    agent = create_agent(model=model,
                        tools=[get_filter_context, search_documents],
                        system_prompt=SYSTEM_PROMPT,
                        checkpointer=memory)
    
    cl.user_session.set('agent', agent)
    cl.user_session.set('thread_id', cl.context.session.id)

    await cl.Message(content="Hello! Upload documents (drag & drop) or ask me a question.").send()



# this will execute when you send a message
@cl.on_message
async def on_message(message: cl.Message): 
    agent = cl.user_session.get('agent')
    thread_id = cl.user_session.get('thread_id')


    if message.elements:
        with tempfile.TemporaryDirectory() as tmpdir:
            for elem in message.elements:
                # process all files here.
                dest = os.path.join(tmpdir, elem.name)
                with open(elem.path, 'rb') as src, open(dest, 'wb') as dst:
                    dst.write(src.read())

            msg = cl.Message(content="Ingesting documents...")
            await msg.send()

            stats = rag.ingest_directory(tmpdir)
            msg.content = f"Files have been ingested. Stats: {stats}"
            await msg.update()

            return


    config = {'configurable': {'thread_id': thread_id}}

    response_msg = cl.Message(content='Thinking...')
    await response_msg.send()

    result = await agent.ainvoke({'messages':[HumanMessage(message.content)]},
                                 config=config)
    
    response_msg.content = result['messages'][-1].text

    await response_msg.update()