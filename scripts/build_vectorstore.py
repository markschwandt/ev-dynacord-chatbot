"""
Vector Store Builder
Reads extracted chunks and builds a ChromaDB vector store using OpenAI embeddings.
"""

import os
import json
import logging
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other vars) from ../.env if present
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("vectorstore.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CHUNKS_FILE = os.environ.get(
    "CHUNKS_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "all_chunks.json"),
)
VECTORSTORE_DIR = os.environ.get(
    "VECTORSTORE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "chromadb"),
)
COLLECTION_NAME = "ev_dynacord_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 500  # text-embedding-3-small accepts up to 2048 inputs per request
RATE_LIMIT_DELAY = 0.1  # seconds between batches (text-embedding-3-small has generous rate limits)


def load_chunks(path: str) -> list[dict]:
    """Load chunks from the JSON file."""
    logger.info(f"Loading chunks from {path}")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def get_embeddings(client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Get embeddings for a batch of texts from OpenAI."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def build_vectorstore(chunks: list[dict]):
    """Build ChromaDB vector store from chunks with OpenAI embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either export it in your shell or "
            "create a .env file at the repo root with: OPENAI_API_KEY=sk-..."
        )
    openai_client = OpenAI(api_key=api_key)

    # Initialize ChromaDB with persistent storage
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # Delete collection if it exists (fresh build)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Electro-Voice & Dynacord product documentation"},
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches")

    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_idx : batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Convert metadata values to strings (ChromaDB requirement)
        clean_metadatas = []
        for m in metadatas:
            clean_metadatas.append({
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in m.items()
            })

        try:
            embeddings = get_embeddings(openai_client, texts)

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=clean_metadatas,
            )

            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"Batch {batch_num}/{total_batches} complete")

        except Exception as e:
            logger.error(f"Error on batch {batch_num}: {e}")
            # Retry once after a longer delay
            time.sleep(5)
            try:
                embeddings = get_embeddings(openai_client, texts)
                collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=clean_metadatas,
                )
                logger.info(f"Batch {batch_num} succeeded on retry")
            except Exception as e2:
                logger.error(f"Batch {batch_num} failed permanently: {e2}")
                continue

        time.sleep(RATE_LIMIT_DELAY)

    logger.info("=" * 60)
    logger.info(f"Vector store built successfully!")
    logger.info(f"  Collection:   {COLLECTION_NAME}")
    logger.info(f"  Documents:    {collection.count()}")
    logger.info(f"  Storage:      {VECTORSTORE_DIR}")
    logger.info("=" * 60)


def main():
    chunks = load_chunks(os.path.abspath(CHUNKS_FILE))
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()
