"""
RAGWire - OpenAI-Compatible FastAPI Server

Setup in OpenWebUI:
  Settings → Connections → URL: http://localhost:8000

Select agent via env var (default: langchain_agent):
  set AGENT=langchain_agent && python main.py
  set AGENT=crewai_agent    && python main.py
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router, agent

app = FastAPI(title="RAGWire OpenAI-Compatible API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(router)

print(f"[RAGWire] Using agent: {agent.MODEL_ID}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
