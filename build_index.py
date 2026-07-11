#!/usr/bin/env python3
"""
build_index.py
==============

Build a FAISS vector database from the Markdown wiki inside ``knowledge_base/``.

Pipeline
--------
1. Load every ``.md`` file from ``knowledge_base/``.
2. Split each document into coherent chunks with LangChain's
   ``RecursiveCharacterTextSplitter`` and attach rich metadata (source file,
   title, chunk id, document index, char offsets) so a downstream bot can cite
   sources.
3. Embed every chunk with the **BGE-M3** model (``BAAI/bge-m3``) via
   ``sentence-transformers``. Embeddings are generated in batches and L2
   normalised so inner-product == cosine similarity.
4. Build a flat FAISS index (``IndexFlatIP``) and persist it to
   ``faiss.index`` together with a companion ``metadata.pkl`` that maps every
   FAISS row id -> chunk metadata dict.
5. Run three example queries end-to-end (embed query -> search FAISS -> print
   top-k chunks) so a human can verify the results are meaningful.

Usage
-----
    python build_index.py                 # build (if missing) + run quality checks
    python build_index.py --rebuild       # force a full rebuild
    python build_index.py --query "Who is Cypher?" --k 5
    python build_index.py --no-check      # build only, skip quality checks

The first run downloads the BGE-M3 weights (~2 GB) from HuggingFace; a friendly
notice is printed while that happens.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Shared prompt-injection security layer (ProtectAI DeBERTa classifier).
# The classifier itself loads lazily, so this import is cheap.
import security

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
KB_DIR: Path = PROJECT_ROOT / "knowledge_base"
INDEX_PATH: Path = PROJECT_ROOT / "faiss.index"
METADATA_PATH: Path = PROJECT_ROOT / "metadata.pkl"

MODEL_NAME: str = "BAAI/bge-m3"
CHUNK_SIZE: int = 800            # ~100-300 words / ~500-1000 tokens
CHUNK_OVERLAP: int = 100
SEPARATORS: List[str] = ["\n\n", "\n", ". ", " "]
EMBED_BATCH_SIZE: int = 32       # batch size for the embedding model (CPU)
GPU_EMBED_BATCH_SIZE: int = 64   # batch size for the embedding model (GPU)
TOP_K: int = 5                   # results to show per quality-check query

# Domain-relevant example queries drawn from the knowledge base.
EXAMPLE_QUERIES: List[str] = [
    "Who is the Professor's greatest enemy?",
    "What is renewal and how does it work?",
    "Describe the Cypher race and their origins.",
]


# --------------------------------------------------------------------------- #
# Small logging helper (flushes so progress is visible during long runs)
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    bar = "=" * 70
    log(f"\n{bar}\n{title}\n{bar}")


# --------------------------------------------------------------------------- #
# Optional-dependency imports with helpful error messages
# --------------------------------------------------------------------------- #
def _import_deps() -> Tuple[Any, Any, Any]:
    """Import faiss, sentence_transformers and langchain splitter.

    Returns (faiss, SentenceTransformer, RecursiveCharacterTextSplitter).
    Exits with a helpful message if a required package is missing/broken.
    """
    try:
        import faiss  # type: ignore
    except Exception:  # pragma: no cover - environment specific
        log("ERROR: could not import 'faiss'. Install it with:\n"
            "    pip install faiss-cpu")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:  # pragma: no cover - environment specific
        log("ERROR: could not import 'sentence_transformers'. Install it with:\n"
            "    pip install sentence-transformers")
        sys.exit(1)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # pragma: no cover - environment specific
        log("ERROR: could not import 'langchain_text_splitters'. Install it with:\n"
            "    pip install langchain-text-splitters")
        sys.exit(1)

    return faiss, SentenceTransformer, RecursiveCharacterTextSplitter


# --------------------------------------------------------------------------- #
# 1. Load documents
# --------------------------------------------------------------------------- #
def extract_title(text: str) -> str:
    """Return the document title = first non-empty line.

    Handles both ``# Title`` H1 headings and plain first lines (which is what
    this wiki uses).
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop a leading markdown H1 hash if present.
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        return stripped
    return "(untitled)"


def load_documents() -> List[Dict[str, Any]]:
    """Load every ``.md`` file in ``knowledge_base/``.

    Returns a list of dicts: {doc_index, source, title, text}.
    """
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {KB_DIR}")

    log(f"[load] Found {len(md_files)} Markdown file(s) in '{KB_DIR.name}/'")

    documents: List[Dict[str, Any]] = []
    for doc_index, path in enumerate(md_files):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            log(f"[load] WARNING: could not read {path.name}: {exc}")
            continue
        title = extract_title(text)
        # Store the path relative to the project root (portable + citable).
        source = path.relative_to(PROJECT_ROOT).as_posix()
        documents.append({
            "doc_index": doc_index,
            "source": source,
            "title": title,
            "text": text,
        })

    log(f"[load] Successfully loaded {len(documents)} document(s).")
    return documents


# --------------------------------------------------------------------------- #
# 2. Chunk documents
# --------------------------------------------------------------------------- #
def compute_offsets(doc_text: str, chunk_texts: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Best-effort (char_start, char_end) offsets for each chunk in the doc.

    Returns a list of (start, end) tuples aligned with ``chunk_texts``.
    Offsets are approximate (overlap may shift them slightly) and fall back to
    ``(None, None)`` if a chunk cannot be located.
    """
    offsets: List[Tuple[Optional[int], Optional[int]]] = []
    search_from = 0
    for ct in chunk_texts:
        needle = ct if ct else ""
        idx = doc_text.find(needle, search_from) if needle else -1
        if idx == -1:
            # Try a left-stripped variant (the splitter may trim separators).
            stripped = needle.lstrip()
            idx = doc_text.find(stripped, search_from) if stripped else -1
            if idx == -1:
                offsets.append((None, None))
                continue
            start, end = idx, idx + len(stripped)
        else:
            start, end = idx, idx + len(ct)
        offsets.append((start, end))
        search_from = end if end is not None else search_from
    return offsets


def chunk_documents(
    documents: List[Dict[str, Any]],
    splitter: Any,
) -> List[Dict[str, Any]]:
    """Split every document into chunks and attach per-chunk metadata."""
    all_chunks: List[Dict[str, Any]] = []
    row_id = 0

    for doc in documents:
        text = doc["text"]
        try:
            chunk_texts = splitter.split_text(text)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"[chunk] WARNING: failed to split {doc['source']}: {exc}")
            continue

        offsets = compute_offsets(text, chunk_texts)

        for chunk_id, (c_text, (c_start, c_end)) in enumerate(zip(chunk_texts, offsets)):
            chunk = {
                "row_id": row_id,            # global FAISS row id
                "doc_index": doc["doc_index"],
                "source": doc["source"],
                "title": doc["title"],
                "chunk_id": chunk_id,        # sequential index within the document
                "text": c_text,
                "char_start": c_start,
                "char_end": c_end,
                "n_chars": len(c_text),
            }
            all_chunks.append(chunk)
            row_id += 1

    log(f"[chunk] Created {len(all_chunks)} chunk(s) from {len(documents)} document(s).")
    if all_chunks:
        avg = sum(c["n_chars"] for c in all_chunks) / len(all_chunks)
        mn = min(c["n_chars"] for c in all_chunks)
        mx = max(c["n_chars"] for c in all_chunks)
        log(f"[chunk] Chunk size stats -> min={mn} avg={avg:.0f} max={mx} chars")
    return all_chunks


# --------------------------------------------------------------------------- #
# 3. Embeddings
# --------------------------------------------------------------------------- #
def get_preferred_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_info(device: str) -> str:
    """Human-readable description of the compute device (incl. GPU/VRAM)."""
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"cuda ({name}, {vram_gb:.1f} GB VRAM)"
        except Exception:
            return "cuda"
    return "cpu"


def load_embedder(SentenceTransformer: Any, device: str = "cpu") -> Any:
    """Load the BGE-M3 sentence-transformer model onto ``device``."""
    log(f"[embed] Loading embedding model '{MODEL_NAME}' onto device "
        f"'{device_info(device)}' ...")
    log("[embed] NOTE: the first run downloads the BGE-M3 weights (~2 GB) "
        "from HuggingFace. Please be patient.")
    t0 = time.time()
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        log(f"[embed] ERROR: failed to load model '{MODEL_NAME}': {exc}")
        log("[embed] Common fixes:\n"
            "  * Check your internet connection (weights are downloaded once).\n"
            "  * pip install -U sentence-transformers huggingface_hub\n"
            "  * If the download is blocked, set HF_ENDPOINT or pre-download the model.")
        raise
    try:
        emb_dim = model.get_embedding_dimension()             # sentence-transformers >=3.x
    except AttributeError:
        emb_dim = model.get_sentence_embedding_dimension()    # older API fallback
    log(f"[embed] Model loaded in {time.time() - t0:.1f}s "
        f"(embedding dim = {emb_dim}).")
    return model


def embed_texts(model: Any, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> "np.ndarray":  # type: ignore[name-defined]
    """Embed a list of texts in batches, returning a float32 (n, d) array.

    Vectors are L2-normalised so that inner-product == cosine similarity.
    If a batch triggers an out-of-memory error, the batch size is halved and
    the offending batch is retried (defensive against limited RAM/VRAM).
    """
    import numpy as np  # local import keeps the top of file import-light

    n = len(texts)
    if n == 0:
        raise ValueError("No texts to embed.")
    log(f"[embed] Encoding {n} chunk(s) in batches of {batch_size} ...")

    vectors: List["np.ndarray"] = []  # type: ignore[name-defined]
    bs = batch_size
    i = 0
    t0 = time.time()
    next_progress = 0

    while i < n:
        batch = texts[i:i + bs]
        try:
            vec = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # cosine via inner product
                show_progress_bar=False,
            )
        except (MemoryError, RuntimeError) as exc:
            new_bs = max(1, bs // 2)
            if new_bs == bs:
                # Cannot shrink further; surface the error.
                log(f"[embed] ERROR: out of memory at chunk {i} with batch_size={bs}: {exc}")
                log("[embed] Tip: lower CHUNK_SIZE or EMBED_BATCH_SIZE at the top of this file.")
                raise
            # Free fragmented allocator memory (especially on CUDA) before retry.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log(f"[embed] WARNING: memory error at chunk {i} "
                f"(batch_size={bs}); retrying with batch_size={new_bs}. {exc}")
            bs = new_bs
            continue  # retry the same `i` with the smaller batch size

        vectors.append(vec)
        i += len(batch)

        # Progress every ~5%.
        pct = int(100 * i / n)
        if pct >= next_progress:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n - i) / rate if rate > 0 else 0.0
            log(f"[embed]   {i}/{n} chunks ({pct}%) "
                f"| {rate:.1f} chunks/s | ETA ~{eta:.0f}s")
            next_progress = pct + 5

    import numpy as _np  # noqa: F811
    mat = _np.vstack(vectors).astype("float32")
    log(f"[embed] Done in {time.time() - t0:.1f}s. "
        f"Embedding matrix shape = {mat.shape}, dtype = {mat.dtype}.")
    return mat


# --------------------------------------------------------------------------- #
# 4. FAISS index build + persistence
# --------------------------------------------------------------------------- #
def build_and_save_index(faiss: Any, embeddings: "np.ndarray",  # type: ignore[name-defined]
                         metadata: List[Dict[str, Any]]) -> Any:
    """Build a flat inner-product FAISS index and persist index + metadata."""
    dim = int(embeddings.shape[1])
    log(f"[index] Building IndexFlatIP (dim={dim}) for "
        f"{embeddings.shape[0]} vectors ...")
    try:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    except Exception as exc:  # pragma: no cover - defensive
        log(f"[index] ERROR: FAISS build failed: {exc}")
        raise

    assert index.ntotal == len(metadata), (
        f"Index size {index.ntotal} != metadata size {len(metadata)}"
    )

    try:
        faiss.write_index(index, str(INDEX_PATH))
    except Exception as exc:  # pragma: no cover - defensive
        log(f"[index] ERROR: could not write '{INDEX_PATH}': {exc}")
        raise

    try:
        with open(METADATA_PATH, "wb") as fh:
            pickle.dump(metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # pragma: no cover - defensive
        log(f"[index] ERROR: could not write '{METADATA_PATH}': {exc}")
        raise

    size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)
    meta_kb = METADATA_PATH.stat().st_size / 1024
    log(f"[index] Saved '{INDEX_PATH.name}' ({size_mb:.2f} MB, "
        f"{index.ntotal} vectors) and '{METADATA_PATH.name}' ({meta_kb:.1f} KB).")
    return index


def load_index_and_metadata(faiss: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load a previously persisted FAISS index and its metadata companion."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Index/metadata not found ({INDEX_PATH.name}, {METADATA_PATH.name}). "
            "Run without --query first to build them."
        )
    log(f"[load] Reading existing '{INDEX_PATH.name}' and '{METADATA_PATH.name}' ...")
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as fh:
        metadata = pickle.load(fh)
    log(f"[load] Loaded index with {index.ntotal} vectors and "
        f"{len(metadata)} metadata records.")
    return index, metadata


# --------------------------------------------------------------------------- #
# 5. Quality check
# --------------------------------------------------------------------------- #
def query_index(model: Any, index: Any, metadata: List[Dict[str, Any]],
                query: str, k: int = TOP_K) -> List[Tuple[float, Dict[str, Any]]]:
    """Embed ``query``, search FAISS, return top-k (score, metadata) pairs."""
    import numpy as np
    qvec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, ids = index.search(qvec, k)
    results: List[Tuple[float, Dict[str, Any]]] = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id == -1:
            continue
        results.append((float(score), metadata[int(row_id)]))
    return results


def print_results(query: str, results: List[Tuple[float, Dict[str, Any]]]) -> None:
    log(f"\n  QUERY: \"{query}\"")
    if not results:
        log("    (no results)")
        return
    for rank, (score, meta) in enumerate(results, start=1):
        snippet = " ".join(meta["text"].split())  # collapse whitespace
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + " ..."
        loc = ""
        if meta.get("char_start") is not None:
            loc = f" [chars {meta['char_start']}-{meta['char_end']}]"
        log(f"    {rank}. score={score:.4f} | {meta['source']} "
            f"| title=\"{meta['title']}\" | chunk_id={meta['chunk_id']}{loc}")
        log(f"       \"{snippet}\"")


def run_quality_checks(model: Any, index: Any,
                       metadata: List[Dict[str, Any]], k: int = TOP_K) -> None:
    section("Quality check — running example queries")
    log(f"Showing top-{k} matches per query (cosine similarity via inner product).")
    for query in EXAMPLE_QUERIES:
        results = query_index(model, index, metadata, query, k=k)
        print_results(query, results)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build(faiss: Any, SentenceTransformer: Any, splitter_cls: Any,
          device: str = "cpu", batch_size: int = EMBED_BATCH_SIZE) -> None:
    """Full build: load -> chunk -> embed -> index -> persist."""
    section("Building FAISS index from knowledge_base/")

    documents = load_documents()

    splitter = splitter_cls(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=False,
        length_function=len,
    )
    chunks = chunk_documents(documents, splitter)

    # --- Change 1: prompt-injection security filter ------------------------ #
    # Scan every chunk with the ProtectAI classifier and drop any detected as
    # a prompt-injection attack so unsafe content never enters the FAISS index.
    section("Security check — filtering prompt-injection chunks")
    log(f"[security] Scanning {len(chunks)} chunk(s) with the ProtectAI "
        f"prompt-injection classifier '{security.INJECTION_MODEL_ID}' ...")
    safe_chunks = security.filter_unsafe_chunks(
        chunks, threshold=security.DEFAULT_THRESHOLD
    )
    rejected = len(chunks) - len(safe_chunks)
    log(f"[security] Rejected {rejected} chunk(s) as prompt injection; "
        f"{len(safe_chunks)} safe chunk(s) will be indexed.")
    chunks = safe_chunks
    if not chunks:
        log("[security] WARNING: no safe chunks remain after filtering — "
            "the resulting index will be empty.")
    # --- end security filter ----------------------------------------------- #

    model = load_embedder(SentenceTransformer, device=device)
    embeddings = embed_texts(model, [c["text"] for c in chunks], batch_size=batch_size)

    build_and_save_index(faiss, embeddings, chunks)
    # Free the embedding matrix + model from RAM before the optional checks.
    del embeddings
    log("[build] Build complete.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector DB from the knowledge_base/ Markdown wiki.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force a full rebuild even if the index exists.")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query against the index (builds it first if missing).")
    parser.add_argument("--k", type=int, default=TOP_K,
                        help=f"Top-k results to show (default: {TOP_K}).")
    parser.add_argument("--no-check", action="store_true",
                        help="After building, skip the example quality-check queries.")
    args = parser.parse_args(argv)

    section("build_index.py — FAISS vector DB builder")
    log(f"Python  : {sys.version.split()[0]} ({sys.executable})")
    log(f"Project : {PROJECT_ROOT}")
    log(f"Model   : {MODEL_NAME} (BGE-M3)")
    log(f"Output  : {INDEX_PATH.name} + {METADATA_PATH.name}")

    faiss, SentenceTransformer, splitter_cls = _import_deps()

    device = get_preferred_device()
    # Larger batches are safe and much faster on GPU; keep 32 for CPU.
    batch_size = GPU_EMBED_BATCH_SIZE if device == "cuda" else EMBED_BATCH_SIZE
    log(f"[main] Compute device: {device_info(device)} | embed batch size: {batch_size}")

    need_build = (
        args.rebuild
        or not INDEX_PATH.exists()
        or not METADATA_PATH.exists()
    )
    if need_build:
        build(faiss, SentenceTransformer, splitter_cls,
              device=device, batch_size=batch_size)
    else:
        log("\n[skip] Existing index found. Use --rebuild to force a rebuild.")

    # Load whatever we have (freshly built or pre-existing).
    index, metadata = load_index_and_metadata(faiss)
    model = load_embedder(SentenceTransformer, device=device)

    if args.query:
        section(f"Single query (k={args.k})")
        results = query_index(model, index, metadata, args.query, k=args.k)
        print_results(args.query, results)
    elif not args.no_check:
        run_quality_checks(model, index, metadata, k=args.k)
    else:
        log("\n[done] Build finished; quality checks skipped (--no-check).")

    section("Done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nInterrupted by user.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - top-level safety net
        log(f"\nFATAL: {exc}")
        log(traceback.format_exc())
        raise SystemExit(1)
