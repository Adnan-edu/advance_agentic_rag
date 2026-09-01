 Run it with:

  ./start_qdrant.sh
  uv run scripts/check_qdrant.py

  Future upgrades use the same command; the volume survives container and image replacement.

  You can inspect it with:

  docker volume inspect qdrant_storage

  To pin a safer specific image version:

  QDRANT_IMAGE=qdrant/qdrant:v1.15.3 ./start_qdrant.sh

----

  ----

    RAGWIRE_MODEL_TIER=free uv run chainlit run app.py \
    --host 127.0.0.1 \
    --port 8000 \
    -w

----


06 FastAPI RAG Backend % AGENT=01_langchain_agent RAGWIRE_MODEL_TIER=free uv run main.py

----
07 Chainlit Chat Frontend

chainlit run app.py --host 127.0.0.1 --port 8000 -w

  cd "/Users/adnan/Documents/adnanedu/udemy/AI/ragwire/ragwire/06 FastAPI RAG Backend"

  QDRANT_API_KEY='' \
  RAGWIRE_MODEL_TIER=paid \
  AGENT=01_langchain_agent \
  ../.venv/bin/python main.py