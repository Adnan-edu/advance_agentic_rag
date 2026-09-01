"""
Main RAG pipeline orchestrating document ingestion and retrieval.

Coordinates all components of the RAG system:
- Document loading and conversion
- Text splitting and chunking
- Metadata extraction
- Embedding generation
- Vector store operations
- Hybrid retrieval

================================================================================
OVERALL OVERVIEW — how this file works
================================================================================
This module defines the `RAGWire` class: the single entry point users interact
with (e.g. `rag = RAGWire("config.yaml")`). It is a *facade/orchestrator* — it
contains no embedding or vector-store logic itself; instead it wires together
every other subpackage of the library based on a YAML config file.

The lifecycle has two phases:

1. INGESTION (write path):
   file → MarkItDownLoader (any format → markdown text)
        → splitter (text → chunks)
        → MetadataExtractor (LLM reads doc, extracts fields like company_name)
        → QdrantStore.add_documents (chunks embedded + stored with metadata)
   Deduplication happens at the file level via SHA-256 hashes, so re-running
   ingestion on the same files is a cheap no-op ("skipped").

2. RETRIEVAL (read path):
   query → (optional) LLM filter extraction ("Apple 2024 report" →
            {"company_name": "apple inc.", "fiscal_year": 2024})
         → Qdrant hybrid search (dense vectors + sparse/BM25) with the
            extracted filters applied as Qdrant payload conditions
         → list of langchain Documents returned to the caller.

Agent-facing helpers (`get_filter_context`, `extract_filters`,
`discover_metadata_fields`, `get_field_values`) expose the collection's
metadata schema and stored values so an LLM agent can build precise filters
before calling `retrieve()`.
================================================================================
"""

# --- Standard library imports ---
import json                                      # parse the LLM's JSON filter output
import logging                                   # module-level logger
from datetime import datetime, timezone          # UTC timestamps stamped on every chunk
from pathlib import Path                         # directory walking in ingest_directory
from typing import Optional, List, Dict, Any, TypedDict  # type hints for public APIs


# Typed shape of one ingestion failure: which file and what went wrong.
class IngestError(TypedDict):
    file: str
    error: str


# Typed shape of the dict returned by ingest_documents()/ingest_directory().
# Gives callers IDE autocompletion on stats["processed"], stats["errors"], etc.
class IngestStats(TypedDict):
    total: int              # how many files were passed in
    processed: int          # successfully chunked + stored
    skipped: int            # already ingested (file hash matched)
    failed: int             # load or processing errors
    chunks_created: int     # total chunks written to the vector store
    errors: List[IngestError]  # one entry per failed file

# LangChain prompt builder — used to drive the filter-extraction LLM call.
from langchain_core.prompts import ChatPromptTemplate

# Import pipeline components — one import per subsystem this class orchestrates.
from .config import Config                                     # YAML config loader w/ ${ENV} expansion
from ..loaders.markitdown_loader import MarkItDownLoader        # any-format → markdown text
from ..processing.splitter import get_splitter, get_markdown_splitter  # chunking strategies
from ..processing.hashing import sha256_file_from_path, sha256_chunk   # dedup hashes
from ..metadata.extractor import MetadataExtractor              # LLM-based metadata extraction
from ..embeddings.factory import get_embedding                  # provider-agnostic embedding factory
from ..vectorstores.qdrant_store import QdrantStore             # Qdrant wrapper (collections, facets)
from ..retriever.hybrid import get_retriever, hybrid_search     # retrieval strategies

# Logger for this module — actual handlers/levels get configured from the
# config file's [logging] section in _initialize_logging().
logger = logging.getLogger(__name__)


class RAGWire:
    """
    Main RAG pipeline for document ingestion and retrieval.

    Orchestrates the complete RAG workflow from document loading
    to vector store ingestion and retrieval.

    Attributes:
        config: Configuration dictionary
        loader: Document loader instance
        splitter: Text splitter instance
        embedding: Embedding model instance
        vectorstore: Qdrant vector store instance
        retriever: Retriever instance

    Example:
        >>> rag = RAGWire("config.yaml")
        >>> rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
        >>> results = rag.retrieve("What is Amazon's revenue?")
    """

    def __init__(self, config_path: str):
        """
        Initialize the RAG pipeline.

        Reads the YAML config, then builds every component in dependency
        order. After this returns, the pipeline is fully ready for both
        ingestion and retrieval.

        Args:
            config_path: Path to configuration YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        logger.info(f"Loading configuration from {config_path}")

        # Load configuration — Config parses the YAML and resolves
        # ${ENV_VAR} placeholders; we keep just the resulting dict.
        self.config = Config(config_path).config

        # Cache for stored filter values — populated on first query, invalidated after ingestion
        # (avoids hitting Qdrant's facet API on every single filter extraction).
        self._stored_values_cache: Optional[Dict[str, Any]] = None

        # Initialize components — order matters:
        # embeddings must exist before the vector store (Qdrant needs the
        # embedding model to embed queries), and the vector store must exist
        # before the retriever (which wraps it).
        self._initialize_logging()      # configure log handlers first so later steps log properly
        self._initialize_loader()       # document → text converter
        self._initialize_splitter()     # text → chunks
        self._initialize_embeddings()   # chunks/queries → vectors
        self._initialize_llm()          # LLM + metadata extractor
        self._initialize_vectorstore()  # Qdrant connection + collection setup
        self._initialize_retriever()    # search wrapper over the vector store

        logger.info("RAG pipeline initialized successfully")

    def _initialize_logging(self) -> None:
        """Apply logging configuration from config file."""
        # Pull the [logging] section; if the user didn't configure logging,
        # leave whatever handlers the host app already set up untouched.
        log_config = self.config.get("logging", {})
        if not log_config:
            return
        # Imported lazily so merely importing this module never reconfigures logging.
        from ..utils.logging import setup_logging, setup_colored_logging
        level = log_config.get("level", "INFO")
        log_file = log_config.get("log_file")  # optional file sink (e.g. ./.log/ragwire.log)
        # Two flavors: ANSI-colored console output, or plain (notebook-friendly).
        if log_config.get("colored", False):
            setup_colored_logging(log_level=level, log_file=log_file)
        else:
            setup_logging(
                log_level=level,
                log_file=log_file,
                console_output=log_config.get("console_output", True),
            )

    def _initialize_loader(self) -> None:
        """Initialize document loader."""
        loader_config = self.config.get("loader", {})
        # MarkItDown converts PDF/DOCX/XLSX/PPTX/etc. into markdown text —
        # one loader handles every supported format.
        self.loader = MarkItDownLoader()
        # Which file extensions ingest_directory() will pick up; configurable
        # via loader.extensions in the YAML.
        self.loader_extensions = loader_config.get(
            "extensions", [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]
        )
        logger.info("Document loader initialized")

    def _initialize_splitter(self) -> None:
        """Initialize text splitter."""
        splitter_config = self.config.get("splitter", {})
        # Large defaults (10k chars, 2k overlap): chunks stay big enough to
        # preserve context for both retrieval and LLM metadata extraction.
        chunk_size = splitter_config.get("chunk_size", 10000)
        chunk_overlap = splitter_config.get("chunk_overlap", 2000)
        strategy = splitter_config.get("strategy", "markdown")

        # "recursive" = generic character splitter; default "markdown" splits
        # on headings first, which suits MarkItDown's markdown output.
        if strategy == "recursive":
            self.splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            self.splitter = get_markdown_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        logger.info(f"Text splitter initialized (strategy={strategy}, chunk_size={chunk_size})")

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        embedding_config = self.config.get("embeddings", {})
        # Embeddings are mandatory — fail fast with a copy-pasteable example
        # instead of a cryptic error deeper in the stack.
        if not embedding_config or not embedding_config.get("provider"):
            raise ValueError(
                "Missing [embeddings] section or embeddings.provider in config.yaml.\n"
                "Example:\n"
                "  embeddings:\n"
                "    provider: ollama\n"
                "    model: nomic-embed-text\n"
                "Valid providers: ollama, openai, huggingface, google, fastembed"
            )
        # Factory returns a LangChain Embeddings object for whichever provider
        # was configured (local HuggingFace, OpenAI API, Ollama, etc.).
        self.embedding = get_embedding(embedding_config)
        logger.info(
            f"Embedding model initialized (provider={embedding_config.get('provider')})"
        )

    def _initialize_llm(self) -> None:
        """Initialize LLM and metadata extractor.

        The LLM serves two jobs in this pipeline:
        1. Extracting document metadata during ingestion (company, year, ...).
        2. Extracting query filters at retrieval time (auto_filter / agent tools).
        """
        llm_config = self.config.get("llm", {})
        if not llm_config:
            raise ValueError("No [llm] section found in config — required for metadata extraction")

        provider = llm_config.get("provider", "ollama")
        model = llm_config.get("model")
        if not model:
            raise ValueError("llm.model must be set in config")
        base_url = llm_config.get("base_url", "http://localhost:11434")  # Ollama default

        # Maps provider → the pip command shown if its optional extra isn't installed.
        _llm_install = {
            "ollama": "pip install langchain-ollama",
            "openai": "pip install \"ragwire[openai]\"",
            "google": "pip install \"ragwire[google]\"",
            "gemini": "pip install \"ragwire[google]\"",
            "groq": "pip install \"ragwire[groq]\"",
            "anthropic": "pip install \"ragwire[anthropic]\"",
        }
        # Each provider branch imports its LangChain integration lazily —
        # users only need the package for the provider they actually use.
        try:
            if provider == "ollama":
                from langchain_ollama import ChatOllama
                extra = {}
                # num_ctx lets users raise Ollama's context window for big docs.
                if "num_ctx" in llm_config:
                    extra["num_ctx"] = llm_config["num_ctx"]
                llm = ChatOllama(model=model, base_url=base_url, **extra)
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                # api_key/base_url only passed when set — a custom base_url is
                # how OpenAI-compatible gateways (e.g. OpenRouter) are supported.
                openai_kwargs = {"model": model}
                if "api_key" in llm_config and llm_config["api_key"]:
                    openai_kwargs["api_key"] = llm_config["api_key"]
                if "base_url" in llm_config and llm_config["base_url"]:
                    openai_kwargs["base_url"] = llm_config["base_url"]
                llm = ChatOpenAI(**openai_kwargs)
            elif provider == "google" or provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=model, google_api_key=llm_config.get("api_key"))
            elif provider == "groq":
                from langchain_groq import ChatGroq
                llm = ChatGroq(model=model, groq_api_key=llm_config.get("api_key"))
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model=model, anthropic_api_key=llm_config.get("api_key"))
            else:
                # Unknown provider string in the config — list the valid ones.
                valid = "ollama, openai, google, groq, anthropic"
                raise ValueError(
                    f"Unsupported LLM provider: '{provider}'. Valid options: {valid}"
                )
        except ImportError:
            # The integration package is missing — tell the user exactly what to install.
            install_cmd = _llm_install.get(provider, f"pip install \"ragwire[{provider}]\"")
            raise ImportError(
                f"Required package for LLM provider '{provider}' is not installed.\n"
                f"Run: {install_cmd}"
            )

        # Wire the LLM into the metadata extractor. A metadata.config_file
        # (e.g. health_metadata.yaml) defines *custom* fields per domain;
        # without one we fall back to the built-in financial schema.
        metadata_config = self.config.get("metadata", {})
        metadata_yaml = metadata_config.get("config_file") if metadata_config else None

        if metadata_yaml:
            self.metadata_extractor = MetadataExtractor.from_yaml(llm, metadata_yaml)
            logger.info(f"Metadata extractor loaded from: {metadata_yaml}")
            # _filter_fields drives every filter-related feature (extraction,
            # agent context, facet lookups) — custom fields if defined, else defaults.
            self._filter_fields = self.metadata_extractor.fields or ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]
        else:
            self.metadata_extractor = MetadataExtractor(llm)
            self._filter_fields = ["company_name", "doc_type", "fiscal_quarter", "fiscal_year"]
        logger.info(f"LLM initialized for metadata extraction (provider={provider}, model={model})")

    def _initialize_vectorstore(self) -> None:
        """Initialize vector store."""
        vectorstore_config = self.config.get("vectorstore", {})
        # A Qdrant URL is mandatory — fail fast with setup instructions.
        if not vectorstore_config or not vectorstore_config.get("url"):
            raise ValueError(
                "Missing [vectorstore] section or vectorstore.url in config.yaml.\n"
                "Example:\n"
                "  vectorstore:\n"
                "    url: http://localhost:6333\n"
                "    collection_name: my_docs\n"
                "Start Qdrant locally with: docker run -p 6333:6333 qdrant/qdrant"
            )
        collection_name = vectorstore_config.get("collection_name", "rag_documents")
        use_sparse = vectorstore_config.get("use_sparse", True)        # hybrid = dense + sparse vectors
        force_recreate = vectorstore_config.get("force_recreate", False)  # wipe + rebuild collection
        self._ingest_batch_size = vectorstore_config.get("ingest_batch_size", 50)  # upsert batch size

        # Wrapper owns the Qdrant client plus collection/facet/index helpers.
        self.vectorstore_wrapper = QdrantStore(
            config=vectorstore_config,
            embedding=self.embedding,
            collection_name=collection_name,
        )

        # Handle collection creation / recreation
        collection_exists = self.vectorstore_wrapper.collection_exists()

        # force_recreate=true: drop the collection so it's rebuilt fresh —
        # needed when switching embedding models (vector dimensions change).
        if force_recreate and collection_exists:
            self.vectorstore_wrapper.delete_collection()
            logger.info(f"Deleted existing collection for recreation: {collection_name}")
            collection_exists = False

        if not collection_exists:
            # New collection — created with sparse vectors if hybrid search is on.
            self.vectorstore_wrapper.create_collection(use_sparse=use_sparse)
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Using existing collection: {collection_name}")

        # The LangChain VectorStore object used for add_documents()/as_retriever().
        self.vectorstore = self.vectorstore_wrapper.get_store(use_sparse=use_sparse)
        # Index file_hash (powers dedup lookups) plus any metadata fields that
        # already exist, so payload filtering and facets stay fast.
        existing_fields = self.vectorstore_wrapper.get_metadata_keys()
        self.vectorstore_wrapper.create_payload_indexes(["file_hash"] + existing_fields)
        logger.info("Vector store initialized")

    def _initialize_retriever(self) -> None:
        """Initialize retriever."""
        retriever_config = self.config.get("retriever", {})
        search_type = retriever_config.get("search_type", "hybrid")  # similarity | mmr | hybrid
        top_k = retriever_config.get("top_k", 5)
        # auto_filter=true makes retrieve() run LLM filter extraction on every
        # query automatically (instead of the caller/agent supplying filters).
        self._auto_filter = retriever_config.get("auto_filter", False)
        # Default retriever holds the baseline search_type/search_kwargs;
        # retrieve() clones these per-call so overrides don't leak.
        self.retriever = get_retriever(
            self.vectorstore, top_k=top_k, search_type=search_type
        )
        logger.info(f"Retriever initialized (type={search_type}, top_k={top_k}, auto_filter={self._auto_filter})")

    def ingest_documents(self, file_paths: List[str]) -> IngestStats:
        """
        Ingest documents into the vector store.

        Metadata is extracted from each document using the configured LLM.

        Per-file flow: hash → dedup check → load → chunk + extract metadata
        → batch-upsert to Qdrant. One file failing never aborts the run; it
        is recorded in stats["errors"] and the loop continues.

        Args:
            file_paths: List of file paths to ingest

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
            >>> print(f"Processed {stats['processed']} documents")
        """
        # Running tallies returned to the caller (see IngestStats).
        stats = {
            "total": len(file_paths),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks_created": 0,
            "errors": [],
        }

        logger.info(f"Starting ingestion of {len(file_paths)} documents")

        # Show a progress bar when tqdm is available; ingestion works fine without it.
        try:
            from tqdm import tqdm
            file_iter = tqdm(file_paths, desc="Ingesting", unit="file")
        except ImportError:
            file_iter = file_paths

        for file_path in file_iter:
            try:
                # File-level deduplication — skip if already ingested
                # (a point with this file_hash already exists in Qdrant).
                file_hash = sha256_file_from_path(file_path)
                if self.vectorstore_wrapper.file_hash_exists(file_hash):
                    logger.info(f"Skipping (already ingested): {file_path}")
                    stats["skipped"] += 1
                    continue

                # Load document — MarkItDown converts it to markdown text.
                result = self.loader.load(file_path)

                # Loader reports failure via a success flag rather than raising —
                # record it and move on to the next file.
                if not result["success"]:
                    stats["failed"] += 1
                    stats["errors"].append(
                        {"file": file_path, "error": result["error"]}
                    )
                    logger.error(f"Failed to load {file_path}: {result['error']}")
                    continue

                # Process document (pass pre-computed hash to avoid re-reading file):
                # split into chunks, LLM-extract metadata, attach it to every chunk.
                chunks = self._process_document(
                    text=result["text_content"],
                    file_path=file_path,
                    file_name=result["file_name"],
                    file_type=result["file_type"],
                    file_hash=file_hash,
                )

                # Add to vector store in batches to avoid write timeouts
                # (each add_documents call embeds + upserts one batch).
                if chunks:
                    batch_size = getattr(self, "_ingest_batch_size", 50)
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]
                        self.vectorstore.add_documents(batch)
                    stats["chunks_created"] += len(chunks)
                    stats["processed"] += 1
                    logger.info(f"Processed {file_path}: {len(chunks)} chunks")

            except Exception as e:
                # Catch-all per file: hashing, loading, splitting, embedding or
                # upsert errors all land here — log and keep going.
                stats["failed"] += 1
                stats["errors"].append({"file": file_path, "error": str(e)})
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)

        # Create payload indexes for all metadata fields so facet API works
        # (new LLM-extracted fields may have appeared during this run).
        all_fields = self.vectorstore_wrapper.get_metadata_keys()
        self.vectorstore_wrapper.create_payload_indexes(all_fields)
        self._stored_values_cache = None  # invalidate after ingestion — new values may exist

        logger.info(
            f"Ingestion complete: {stats['processed']}/{stats['total']} documents"
        )
        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> IngestStats:
        """
        Ingest all supported documents from a directory.

        Convenience wrapper: globs the directory for supported files, then
        delegates to ingest_documents().

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories (default: False)
            extensions: File extensions to include (defaults to loader config)

        Returns:
            Dictionary with ingestion statistics

        Example:
            >>> stats = rag.ingest_directory("data/", recursive=True)
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Use explicit extensions if given, else the loader's configured list.
        exts = extensions or self.loader_extensions
        pattern = "**/*" if recursive else "*"  # ** descends into subdirectories

        # Collect files whose extension (case-insensitive) is supported.
        file_paths = [
            str(p) for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        ]

        # Nothing matched — warn and return an empty (but well-formed) stats dict.
        if not file_paths:
            logger.warning(f"No supported files found in {directory} (extensions: {exts})")
            return {
                "total": 0, "processed": 0, "skipped": 0,
                "failed": 0, "chunks_created": 0, "errors": [],
            }

        logger.info(f"Found {len(file_paths)} file(s) in {directory}")
        return self.ingest_documents(file_paths)

    def _process_document(
        self,
        text: str,
        file_path: str,
        file_name: str,
        file_type: str,
        file_hash: str,
    ) -> List[Any]:
        """
        Process a single document into chunks with LLM-extracted metadata.

        Metadata is extracted once from the first chunk using the LLM,
        then attached to every chunk of the document.

        Args:
            text: Document text content
            file_path: Original file path
            file_name: Original file name
            file_type: File type
            file_hash: Pre-computed SHA256 hash of the file

        Returns:
            List of Document objects with metadata
        """
        from langchain_core.documents import Document

        # Split first so we can pass the first chunk to the LLM
        chunk_texts = self.splitter.split_text(text)

        # Extract metadata once from the full document text. extract() builds a
        # compact sample (3k-char prefix + markdown heading outline of the whole
        # doc) and escalates to a 12k window if required fields come back null.
        # Using chunk_texts[0] (~1000 chars) was too little context to reliably find all fields
        llm_metadata = {}
        if chunk_texts:
            try:
                llm_metadata = self.extract_metadata(text)
                logger.debug(f"LLM metadata for {file_name}: {llm_metadata}")
            except Exception as e:
                # Metadata is best-effort: a failed extraction must not block
                # ingestion — the chunks still go in with system metadata only.
                logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")

        # Wrap each chunk in a LangChain Document carrying two metadata layers:
        # system fields (source, hashes, indices) + LLM-extracted domain fields.
        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            # Deterministic chunk identity: file hash + position, hashed with
            # the content — enables chunk-level change detection.
            chunk_id = f"{file_hash}_{i}"
            chunk_hash = sha256_chunk(chunk_id, chunk_text)

            chunk_metadata = {
                "source": file_path,                # original path (used in citations)
                "file_name": file_name,
                "file_type": file_type,
                "file_hash": file_hash,             # file-level dedup key
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "chunk_index": i,                   # position within the document
                "total_chunks": len(chunk_texts),
                "created_at": datetime.now(timezone.utc).isoformat(),
                **llm_metadata,                     # domain fields (company_name, year, ...)
            }

            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents

    @property
    def filter_fields(self) -> List[str]:
        """Return the metadata fields used for filtering and auto-filter extraction.

        These are the semantic/LLM-extracted fields only (e.g. company_name, doc_type,
        fiscal_year). System fields like file_hash, chunk_id, source are excluded.
        Use this instead of discover_metadata_fields() when building filter prompts.
        """
        return self._filter_fields

    @property
    def _stored_values(self) -> Dict[str, Any]:
        """Return cached stored filter values, fetching from Qdrant if needed.

        Lazy + cached: the facet query runs at most once per ingestion cycle
        (the cache is cleared at the end of ingest_documents()).
        """
        if self._stored_values_cache is None:
            self._stored_values_cache = self.vectorstore_wrapper.get_field_values(
                self._filter_fields, limit=50
            )
        return self._stored_values_cache

    def extract_filters(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata filters from a natural language query.

        Returns the raw extracted filters so the caller (e.g. an agent) can
        inspect, adjust, or discard them before passing to retrieve().

        Args:
            query: Natural language query string

        Returns:
            Dict of extracted filters, or None if nothing was extracted.

        Example:
            >>> filters = rag.extract_filters("muscle building studies from 2023")
            >>> # {"research_focus": "muscle building", "publication_year": 2023}
            >>> # Agent inspects and adjusts if needed
            >>> results = rag.retrieve(query, filters=filters)
        """
        # Thin public alias over the private extractor (kept private so the
        # auto_filter path can evolve independently).
        return self._extract_filters_from_query(query)

    def get_filter_context(self, query: str, limit: int = 50) -> str:
        """
        Build a ready-made prompt block for an agent describing available metadata
        filters, their stored values, and the filters extracted from the current query.

        Append or prepend this to your agent's task prompt so the agent can decide
        whether to apply, adjust, or discard the extracted filters before calling retrieve().

        Args:
            query: Natural language query string
            limit: Max stored values to show per field (default: 50)

        Returns:
            Formatted markdown string ready to inject into an agent prompt.

        Example:
            >>> context = rag.get_filter_context("muscle building studies from 2023")
            >>> agent_prompt = context + "\\n\\n" + your_task_prompt
        """
        # Gather the two ingredients: what's actually stored per field, and
        # what the LLM thinks the query is asking to filter on.
        stored_values = self.get_field_values(self._filter_fields, limit=limit)
        extracted = self.extract_filters(query) or {}

        # Section 1: the collection's filterable fields with their real values
        # — lets the agent sanity-check extracted filters against reality.
        lines = ["## RAGWire Filter Context", ""]
        lines.append("### Available Metadata Fields and Stored Values")
        for field in self._filter_fields:
            values = stored_values.get(field, [])
            lines.append(f"- **{field}**: {values}")

        # Section 2: the filters extracted from this particular query.
        lines.append("")
        lines.append("### Extracted Filters from Query")
        if extracted:
            for k, v in extracted.items():
                lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append("- *(no filters extracted)*")

        # Section 3: instructions telling the agent how to use the above.
        lines += [
            "",
            "### Instructions",
            "1. Review the extracted filters above.",
            "2. If an extracted value does not match or closely relate to any stored value, adjust or drop that filter.",
            "3. If the query has no clear metadata intent, pass an empty dict `{}` as filters.",
            "4. Pass the final filters dict to the retrieval tool as `filters=`.",
        ]

        return "\n".join(lines)

    def _extract_filters_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Use the configured LLM to extract metadata filters from a natural language query.

        Passes actual stored values to the LLM so it can match exactly what's in
        the collection — avoids mismatches like 'apple' vs 'apple inc.'.
        """
        # Render each filter field with its real stored values, e.g.
        #   company_name: ['apple inc.', 'microsoft']
        # so the LLM can mirror exact casing/format.
        stored_values = self._stored_values
        fields_desc = "\n".join(
            f"  {field}: {stored_values.get(field, [])}"
            for field in self._filter_fields
        )

        # The prompt teaches the LLM to: extract only explicit references,
        # normalize aliases to stored values, copy the stored format/type,
        # and return strict JSON. ({{ }} are literal braces — LangChain
        # templates reserve single braces for variables like {query}.)
        prompt_template = (
            "You are a metadata filter extractor for a document retrieval system.\n\n"
            "## Task\n"
            "Extract metadata filters as a JSON object from the user query.\n"
            "The filters will be used to narrow down document search results.\n\n"
            "## Rules\n"
            "1. Extract a field only when the query clearly and explicitly refers to it.\n"
            "2. Always extract the value the user asked for — but first check if it is an alias, brand name, or subsidiary of a stored value.\n"
            "   If the extracted value refers to the same real-world entity as a stored value (e.g. 'google' → 'alphabet inc.', 'instagram' → 'meta'), use the stored value instead.\n"
            "   If no stored value matches, extract exactly what the user said.\n"
            "3. Learn the format and structure from stored values, then apply that same format to what the user asked for:\n"
            "   - Casing: if stored values are lowercase, output lowercase.\n"
            "   - Prefixes/suffixes: if stored values use a prefix (e.g. 'q1', 'v2', 'dept-hr'), apply it.\n"
            "   - Data type: if stored values are integers, output integers; if strings, output strings.\n"
            "   - Lists: if stored values are lists (e.g. [2024, 2025]), output a list.\n"
            "4. When a query asks for multiple values of the same field (e.g. '2023 and 2024'), output them as a list.\n"
            "5. Do not infer or guess filters that are not clearly mentioned in the query.\n"
            "6. Return {{}} if the query contains no metadata references at all.\n\n"
            "## Format Examples from Stored Values (not a whitelist)\n"
            f"{fields_desc}\n\n"
            "## Examples\n"
            "- Stored: fiscal_quarter: ['q1','q2','q3'] | Query: 'show me Q4 reports' → {{\"fiscal_quarter\": \"q4\"}}\n"
            "- Stored: fiscal_year: [2024, 2025]       | Query: 'documents from 2022'  → {{\"fiscal_year\": 2022}}\n"
            "- Stored: department: ['engineering']     | Query: 'HR policies'          → {{\"department\": \"hr\"}}\n"
            "- Stored: language: ['en']                | Query: 'French documents'     → {{\"language\": \"fr\"}}\n"
            "- Stored: status: ['active']              | Query: 'all documents'        → {{}}\n"
            "- Stored: company_name: ['alphabet inc.'] | Query: 'google earnings'       → {{\"company_name\": \"alphabet inc.\"}}\n\n"
            "## User Query\n"
            "{query}\n\n"
            "## Output (JSON only, no explanation)\n"
        )

        try:
            # LCEL chain: prompt template piped into the same LLM used for
            # metadata extraction; invoke fills in {query}.
            chain = ChatPromptTemplate.from_template(prompt_template) | self.metadata_extractor.llm
            response = chain.invoke({"query": query})
            text = response.text.strip()
            # Tolerant JSON parsing: find the first '{' and raw_decode from
            # there — survives chatty models that wrap JSON in prose/markdown.
            start = text.find("{")
            if start != -1:
                filters, _ = json.JSONDecoder().raw_decode(text, start)
                if filters:
                    # Normalize all string values (incl. inside lists) to
                    # lowercase to match how stored values are kept.
                    filters = {
                        k: [i.lower() if isinstance(i, str) else i for i in v] if isinstance(v, list)
                           else v.lower() if isinstance(v, str) else v
                        for k, v in filters.items()
                    }
                    logger.info(f"Auto-extracted filters from query: {filters}")
                    return filters
        except Exception as e:
            # Filter extraction is best-effort — on any failure fall back to
            # unfiltered search rather than breaking retrieval.
            logger.warning(f"Auto filter extraction failed: {e}")
        return None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Retrieve documents for a query.

        Main read path: resolve top_k → (optionally) auto-extract filters →
        build a per-call retriever with those settings → run the search.

        Args:
            query: Search query string
            top_k: Number of results (uses config default if not provided)
            filters: Optional metadata filters

        Returns:
            List of retrieved documents

        Example:
            >>> results = rag.retrieve("Amazon Q1 2024 revenue")
            >>> for doc in results:
            ...     print(doc.page_content)
        """
        # Fall back to the configured top_k when the caller didn't override it.
        if top_k is None:
            top_k = self.config.get("retriever", {}).get("top_k", 5)

        # auto_filter mode: derive filters from the query via the LLM —
        # but only when the caller didn't supply explicit ones.
        if filters is None and self._auto_filter:
            filters = self._extract_filters_from_query(query)

        # Build search kwargs without mutating the shared retriever
        # (a fresh retriever per call keeps concurrent queries independent).
        search_kwargs = {**self.retriever.search_kwargs, "k": top_k}
        if filters:
            # Convert the plain dict into Qdrant Filter conditions.
            search_kwargs["filter"] = self._build_qdrant_filter(filters)

        retriever = self.vectorstore.as_retriever(
            search_type=self.retriever.search_type,
            search_kwargs=search_kwargs,
        )
        results = retriever.invoke(query)  # embeds the query + searches Qdrant
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")

        return results

    def hybrid_search(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Perform hybrid search (dense + sparse).

        Direct shortcut to the hybrid strategy, regardless of the configured
        retriever.search_type. Same auto-filter behavior as retrieve().

        Args:
            query: Search query
            k: Number of results
            filters: Optional metadata filters

        Returns:
            List of retrieved documents
        """
        if filters is None and self._auto_filter:
            filters = self._extract_filters_from_query(query)
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
        # Delegate to the standalone hybrid_search helper from retriever.hybrid.
        return hybrid_search(self.vectorstore, query, k=k, filters=qdrant_filter)

    @staticmethod
    def _build_qdrant_filter(filters: Dict[str, Any]) -> Any:
        """Convert a plain dict of metadata filters to a Qdrant Filter object.

        Semantics: AND across fields, OR within a field's list of values.
        e.g. {"company": "apple", "year": [2023, 2024]}
             → company == apple AND (year == 2023 OR year == 2024)
        """
        # Imported lazily: qdrant models are only needed when filters are used.
        from qdrant_client.http import models as rest

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # OR logic within a field: doc must match any one of the values
                # (e.g. fiscal_year [2023, 2024] → year is 2023 OR 2024)
                conditions.append(
                    rest.Filter(
                        should=[
                            rest.FieldCondition(
                                # LangChain's Qdrant store nests payload under
                                # "metadata.", hence the key prefix.
                                key=f"metadata.{key}",
                                match=rest.MatchValue(value=v),
                            )
                            for v in value
                        ]
                    )
                )
            else:
                # Scalar value → simple exact-match condition.
                conditions.append(
                    rest.FieldCondition(
                        key=f"metadata.{key}",
                        match=rest.MatchValue(value=value),
                    )
                )
        # must = AND of all per-field conditions.
        return rest.Filter(must=conditions)

    def discover_metadata_fields(self) -> List[str]:
        """
        Return all metadata field names present in the collection.

        Scrolls a single point from Qdrant to inspect its payload keys.
        Fast — one network call regardless of collection size.

        Returns:
            List of metadata field names, or empty list if collection is empty

        Example:
            >>> fields = rag.discover_metadata_fields()
            >>> print(fields)
            ['company_name', 'doc_type', 'fiscal_year', 'file_name', ...]
        """
        # Pure delegation — the wrapper knows how to introspect the collection.
        return self.vectorstore_wrapper.get_metadata_keys()

    def get_field_values(
        self,
        fields: Any,
        limit: int = 50,
    ) -> Any:
        """
        Return unique values for one or more metadata fields.

        Uses Qdrant's facet API — fast and exact regardless of collection size.
        Creates a payload index on each field automatically if one doesn't exist.

        Args:
            fields: A field name (str) or list of field names
            limit: Max unique values to return per field (default: 50)

        Returns:
            - If fields is a str: list of unique values for that field
            - If fields is a list: dict mapping field name → list of unique values

        Example:
            >>> rag.get_field_values("company_name")
            ['apple', 'microsoft', 'google']

            >>> rag.get_field_values(["company_name", "doc_type"])
            {'company_name': ['apple', 'microsoft'], 'doc_type': ['10-k', '10-q']}
        """
        # Accept either a single field name or a list; normalize to a list for
        # the wrapper call, then unwrap the result to match the input shape.
        single = isinstance(fields, str)
        field_list = [fields] if single else fields
        result = self.vectorstore_wrapper.get_field_values(field_list, limit=limit)
        return result[fields] if single else result

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text using the configured LLM.

        Automatically passes stored collection values so the LLM reuses
        existing entity names (e.g. 'apple inc.') instead of extracting
        inconsistent variants ('apple', 'Apple Inc.').

        Args:
            text: Document text to extract metadata from

        Returns:
            Dictionary of extracted metadata fields

        Example:
            >>> metadata = rag.extract_metadata(open("report.pdf.txt").read())
            >>> print(metadata)
            {'company_name': 'apple inc.', 'doc_type': '10-k', 'fiscal_year': [2025]}
        """
        # stored_values keeps extraction consistent with what's already in the
        # collection (same trick as query-filter extraction).
        return self.metadata_extractor.extract(text, stored_values=self._stored_values)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        collection_info = self.vectorstore_wrapper.get_collection_info()

        # Vector size lives in different places depending on collection layout:
        # a single unnamed vector has .size directly; hybrid collections use
        # named vectors (a dict), so read the size off the first entry.
        vectors = collection_info.config.params.vectors
        if hasattr(vectors, "size"):
            vector_size = vectors.size
        else:
            # Named vectors — take the first one
            vector_size = next(iter(vectors.values())).size

        return {
            "collection_name": self.vectorstore_wrapper.collection_name,
            "total_documents": collection_info.points_count or 0,   # actually point/chunk count
            "vector_size": vector_size,                              # embedding dimension
            "indexed": getattr(collection_info, "indexed_vectors_count", None) or 0,
        }
