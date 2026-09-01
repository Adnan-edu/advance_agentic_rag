"""Run this once to create the SQLite database schema for Chainlit chat history."""

import asyncio
import os
import aiosqlite

DB_PATH = "data/chat_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    identifier TEXT NOT NULL UNIQUE,
    "createdAt" TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    "createdAt" TEXT,
    name TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    tags TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY ("userId") REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "disableFeedback" INTEGER NOT NULL DEFAULT 0,
    streaming INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER,
    "isError" INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT,
    input TEXT,
    output TEXT,
    "createdAt" TEXT,
    start TEXT,
    "end" TEXT,
    "showInput" TEXT,
    language TEXT,
    indent INTEGER,
    generation TEXT,
    "defaultOpen" INTEGER,
    "autoCollapse" INTEGER,
    FOREIGN KEY ("threadId") REFERENCES threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    id TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    value INTEGER NOT NULL,
    comment TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    id TEXT PRIMARY KEY,
    "threadId" TEXT,
    type TEXT,
    url TEXT,
    "chainlitKey" TEXT,
    name TEXT NOT NULL,
    display TEXT,
    "objectKey" TEXT,
    size TEXT,
    page INTEGER,
    language TEXT,
    "forId" TEXT,
    mime TEXT,
    props TEXT
);
"""


async def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    asyncio.run(init())
