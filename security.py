#!/usr/bin/env python3
"""
security.py
===========

Shared **prompt-injection security layer** used by ``build_index.py`` and
``rag_bot.py``.

It wraps the **ProtectAI DeBERTa-v3-small Prompt Injection v2** model from
Hugging Face (``protectai/deberta-v3-small-prompt-injection-v2``), a
text-classification model that labels a piece of text as either ``INJECTION``
or ``SAFE`` together with a confidence score.

The (heavy) ``transformers`` pipeline is created lazily and cached at module
level, so it is loaded exactly once per process and reused for every subsequent
check — keeping indexing and retrieval cheap.

Public API
----------
- :func:`load_classifier`        — explicitly load (and cache) the pipeline.
- :func:`classify`               — return ``(label, score)`` for a text.
- :func:`is_unsafe`              — ``True`` if text is a prompt injection
                                   above ``threshold``.
- :func:`filter_unsafe_chunks`   — filter a list of chunk dicts
                                   (used by ``build_index.py``).
- :func:`filter_unsafe_results`  — filter retrieved ``(score, chunk)`` tuples
                                   (used by ``rag_bot.py``).
- :data:`INJECTION_MODEL_ID`, :data:`DEFAULT_THRESHOLD`, :data:`INJECTION_LABEL`
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Hugging Face model id for the ProtectAI prompt-injection classifier.
INJECTION_MODEL_ID: str = "protectai/deberta-v3-base-prompt-injection-v2"

# A text is considered UNSAFE when the model predicts ``INJECTION`` with a
# confidence ``score >= DEFAULT_THRESHOLD``. 0.5 is the value recommended for
# this task; override it per-call via the ``threshold`` argument if needed.
DEFAULT_THRESHOLD: float = 0.99

# Label the model emits when it detects a prompt-injection attack.
INJECTION_LABEL: str = "INJECTION"

# Maximum tokens fed to the classifier (matches the model's context window).
MAX_LENGTH: int = 512

# How many characters of a rejected chunk's text to include in warning logs.
LOG_PREVIEW_LEN: int = 80


# --------------------------------------------------------------------------- #
# Small logging helper — mirrors the project's print-with-flush style
# (build_index.py / rag_bot.py both use ``log()``). Prefixed with ``[security]``
# so security actions are easy to spot in the output.
# --------------------------------------------------------------------------- #
def _log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Lazy, cached model pipeline
# --------------------------------------------------------------------------- #
# Cached transformers pipeline (created once on first use). ``None`` means
# "not loaded yet".
_classifier: Any = None


def load_classifier() -> Any:
    """Load and cache the ProtectAI prompt-injection classifier pipeline.

    The (heavy) pipeline is created only on the first call and reused for every
    subsequent call. On first load a friendly notice is printed because the
    weights may need to be downloaded from Hugging Face.

    Raises:
        ImportError:  if the ``transformers`` package is not installed.
        RuntimeError: if the model fails to load/download.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    _log(f"[security] Loading prompt-injection classifier "
         f"'{INJECTION_MODEL_ID}' ...")
    _log("[security] NOTE: the first run downloads the model weights from "
         "HuggingFace. Please be patient.")

    t0 = time.time()
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            f"[security] could not import 'transformers': {exc}\n"
            "    Install it with:  pip install transformers torch"
        ) from exc

    try:
        _classifier = pipeline(
            "text-classification",
            model=INJECTION_MODEL_ID,
            truncation=True,
            max_length=MAX_LENGTH,
        )
    except Exception as exc:
        # Reset so a later retry can attempt the load again.
        _classifier = None
        raise RuntimeError(
            f"[security] failed to load model '{INJECTION_MODEL_ID}': {exc}\n"
            "    Common fixes:\n"
            "      * Check your internet connection (weights download once).\n"
            "      * pip install -U transformers huggingface_hub\n"
            "      * If the download is blocked, set HF_ENDPOINT or "
            "pre-download the model."
        ) from exc

    _log(f"[security] Classifier loaded in {time.time() - t0:.1f}s.")
    return _classifier


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
def classify(text: str) -> Tuple[str, float]:
    """Return ``(label, score)`` for ``text``.

    The pipeline returns a single-element list like
    ``[{"label": "INJECTION", "score": 0.998}]``. This helper unpacks it and
    tolerates empty / odd inputs by returning ``("SAFE", 0.0)``.
    """
    if not text or not text.strip():
        return "SAFE", 0.0
    pipe = load_classifier()
    output = pipe(text)
    if isinstance(output, list) and output:
        item = output[0]
        return str(item.get("label", "SAFE")), float(item.get("score", 0.0))
    return "SAFE", 0.0


def is_unsafe(text: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Return ``True`` if ``text`` is detected as a prompt injection.

    A text is **unsafe** when the predicted label is ``INJECTION`` **and** its
    confidence ``score`` is ``>= threshold``.
    """
    label, score = classify(text)
    return label == INJECTION_LABEL and score >= threshold


# --------------------------------------------------------------------------- #
# Bulk filters (used by build_index.py and rag_bot.py respectively)
# --------------------------------------------------------------------------- #
def filter_unsafe_chunks(
    chunks: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Return only the safe chunk dicts, logging each rejected chunk.

    ``chunks`` is the list of chunk dicts produced by ``build_index.py``'s
    ``chunk_documents`` (each expected to carry the chunk text under
    ``text_key``, plus ``source`` and ``chunk_id`` metadata used in the warning
    log). Unsafe chunks are skipped with a ``[security] WARNING`` line.
    """
    safe: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get(text_key, "") or "")
        label, score = classify(text)
        if label == INJECTION_LABEL and score >= threshold:
            _log(f"[security] WARNING: rejected unsafe chunk "
                 f"({label}, {score:.3f}) | "
                 f"source={chunk.get('source', '?')} | "
                 f"chunk_id={chunk.get('chunk_id', '?')} | "
                 f"preview={text.strip()[:LOG_PREVIEW_LEN]!r}")
            continue
        safe.append(chunk)
    return safe


def filter_unsafe_results(
    results: List[Tuple[float, Dict[str, Any]]],
    threshold: float = DEFAULT_THRESHOLD,
    text_key: str = "text",
) -> List[Tuple[float, Dict[str, Any]]]:
    """Filter retrieved ``(score, chunk_dict)`` tuples, removing unsafe chunks.

    ``results`` is the list returned by ``rag_bot.py``'s ``retrieve`` — each
    item is a ``(similarity_score, metadata_dict)`` tuple where the metadata
    dict carries the chunk text under ``text_key``. Unsafe chunks are dropped
    with a ``[security] WARNING`` line.
    """
    safe: List[Tuple[float, Dict[str, Any]]] = []
    for sim_score, meta in results:
        text = str(meta.get(text_key, "") or "")
        label, score = classify(text)
        if label == INJECTION_LABEL and score >= threshold:
            _log(f"[security] WARNING: removed unsafe retrieved chunk "
                 f"({label}, {score:.3f}) | "
                 f"source={meta.get('source', '?')} | "
                 f"chunk_id={meta.get('chunk_id', '?')} | "
                 f"preview={text.strip()[:LOG_PREVIEW_LEN]!r}")
            continue
        safe.append((sim_score, meta))
    return safe
