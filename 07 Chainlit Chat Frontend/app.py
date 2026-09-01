from dotenv import load_dotenv
load_dotenv("../.env")

import io
import json
import os
import re
from typing import Optional

import httpx
import markdown2
from xhtml2pdf import pisa
import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

# ── Config ────────────────────────────────────────────────────────────────────

_STATUS_LINE = re.compile(r"`\[[^\]]+\s+working\.\.\.\]`")

# Strip a leading ```json ... ``` metadata block that some backend responses prepend
_JSON_PREAMBLE = re.compile(r"^\s*```json\s*\{[^`]*?\}\s*```\s*", re.DOTALL)

# Strip one or more bare JSON objects (e.g. filter dicts) from the start of the response
_BARE_JSON = re.compile(r"^\s*(?:\{[^{}]*\}\s*)+")

_WORKING_MESSAGE = "🔎 Searching documents and generating a response…"

def clean_display(text: str) -> str:
    """Remove leading JSON metadata so only the markdown answer is shown."""
    text = _JSON_PREAMBLE.sub("", text)
    text = _BARE_JSON.sub("", text)
    return text.lstrip()

def md_to_pdf(text: str) -> bytes:
    cleaned = "\n".join(
        line for line in text.splitlines()
        if not _STATUS_LINE.search(line)
    ).strip()
    body = markdown2.markdown(cleaned, extras=["tables", "fenced-code-blocks", "cuddled-lists"])
    html = f"""<html><head><meta charset="utf-8"><style>
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.25; padding: 15px; }}
        h1 {{ font-size: 21px; margin: 8px 0 4px 0; line-height: 1.2; }}
        h2 {{ font-size: 18px; margin: 6px 0 3px 0; line-height: 1.2; }}
        h3 {{ font-size: 17px; margin: 5px 0 3px 0; line-height: 1.2; }}
        p {{ margin: 3px 0; line-height: 1.25; }}
        ul, ol {{ margin: 2px 0; padding-left: 20px; }}
        li {{ margin: 1px 0; padding: 0; line-height: 1.25; }}
        li p {{ display: inline; margin: 0; }}
        strong {{ font-weight: bold; }}
        code {{ background: #f4f4f4; padding: 2px 4px; font-size: 13px; }}
        pre {{ background: #f4f4f4; padding: 6px; margin: 4px 0; font-size: 13px; line-height: 1.2; }}
        table {{ border-collapse: collapse; width: 100%; margin: 5px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 4px; font-size: 13px; line-height: 1.2; }}
        th {{ background: #f0f0f0; font-weight: bold; }}
    </style></head><body>{body}</body></html>"""
    buf = io.BytesIO()
    pisa.CreatePDF(html, dest=buf)
    return buf.getvalue()

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8080")
API_KEY = os.getenv("API_KEY", "")
_AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# ── Data layer + Auth ─────────────────────────────────────────────────────────

cl_data._data_layer = SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///data/chat_history.db")

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username == os.getenv("APP_USER", "admin") and password == os.getenv("APP_PASSWORD", "admin"):
        return cl.User(identifier=username, metadata={"role": "user"})
    return None

# ── Chainlit handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])
    cl.user_session.set("last_response_msg", None)
    await cl.Message(content="Hello! Upload documents (drag & drop) or ask me a question.").send()


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    # Rebuild history from previous steps
    history = []
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            history.append({"role": "user", "content": step.get("output", "")})
        elif step.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": step.get("output", "")})
    cl.user_session.set("history", history)


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])

    # ── File upload → FastAPI /upload ─────────────────────────────────────────
    if message.elements:
        handles = [open(elem.path, "rb") for elem in message.elements]
        files = [("files", (elem.name, fh)) for elem, fh in zip(message.elements, handles)]
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                msg = cl.Message(content="Ingesting documents...")
                await msg.send()
                resp = await client.post(f"{API_URL}/upload", files=files, headers=_AUTH_HEADERS)
                resp.raise_for_status()
                msg.content = resp.json()["message"]
                await msg.update()
        finally:
            for fh in handles:
                fh.close()
        return

    # ── Chat → FastAPI /v1/chat/completions (streaming) ───────────────────────
    history.append({"role": "user", "content": message.content})

    response_msg = cl.Message(content=_WORKING_MESSAGE)
    await response_msg.send()

    full_response = ""
    streaming_started = False
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/v1/chat/completions",
                json={"messages": history, "stream": True},
                headers=_AUTH_HEADERS,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        if not streaming_started:
                            response_msg.content = ""
                            await response_msg.update()
                            streaming_started = True
                        full_response += token
                        await response_msg.stream_token(token)
    except httpx.HTTPError:
        response_msg.content = (
            f"⚠️ The backend at `{API_URL}` is unavailable. "
            "Confirm that the lesson 06 FastAPI server is running, then try again."
        )
        response_msg.actions = []
        await response_msg.update()
        return

    # Remove download button from previous response
    prev_msg = cl.user_session.get("last_response_msg")
    if prev_msg:
        prev_msg.actions = []
        await prev_msg.update()

    display_response = clean_display(full_response)
    response_msg.content = display_response
    response_msg.actions = [
        cl.Action(name="download_pdf", payload={"text": display_response}, label="Download PDF", icon="download")
    ]
    await response_msg.update()
    cl.user_session.set("last_response_msg", response_msg)

    history.append({"role": "assistant", "content": display_response})
    cl.user_session.set("history", history)


@cl.action_callback("download_pdf")
async def download_pdf(action: cl.Action):
    pdf_bytes = md_to_pdf(action.payload["text"])
    await cl.Message(
        content="",
        elements=[cl.File(name="response.pdf", content=pdf_bytes, mime="application/pdf")],
    ).send()
