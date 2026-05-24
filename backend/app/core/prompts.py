"""
Prompt versioning system.

All prompts stored in a versioned registry with metadata.
Supports prompt_hash for reproducibility.
Pipeline logs prompt_version per request; full template logged only in DEBUG mode.
"""
import hashlib
from app.config import PROMPT_VERSION


def _hash_prompt(template: str) -> str:
    """Deterministic hash for prompt reproducibility."""
    return hashlib.sha256(template.encode()).hexdigest()[:12]


# --- Prompt Registry ---
# Each version: template, description, created_at
PROMPT_REGISTRY: dict[str, dict] = {
    "v1": {
        "template": (
            "You are a precise question-answering assistant. Follow these rules strictly:\n\n"
            "1. Answer ONLY based on the provided context below. Do not use prior knowledge.\n"
            "2. Cite your sources using the format [Source: <filename>, Page: <page>] after each claim.\n"
            "3. If the context does not contain enough information to answer the question, respond with:\n"
            '   "I don\'t have enough information in the provided documents to answer this question."\n'
            "4. Be concise and factual. Do not speculate or add information not present in the context.\n"
            "5. If multiple sources support a claim, cite all of them."
        ),
        "description": "Default grounded QA prompt with citation enforcement",
        "created_at": "2026-04-18",
    },
    "v2_strict": {
        "template": (
            "You are a precise question-answering assistant operating in STRICT mode.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the provided context. Zero tolerance for external knowledge.\n"
            "2. Every factual claim MUST have a citation: [Source: <filename>, Page: <page>]\n"
            "3. If the context is insufficient, respond EXACTLY with:\n"
            '   "I don\'t have enough information in the provided documents to answer this question."\n'
            "4. Do NOT paraphrase extensively. Stay close to source language.\n"
            "5. If sources conflict, acknowledge the conflict and cite both.\n"
            "6. Maximum response length: 300 words."
        ),
        "description": "Strict mode with tighter citation rules and length limit",
        "created_at": "2026-04-27",
    },
}

# Compute hashes on module load
for version, entry in PROMPT_REGISTRY.items():
    entry["prompt_hash"] = _hash_prompt(entry["template"])


def get_prompt(version: str) -> dict:
    """
    Get a specific prompt version.
    Returns dict with template, description, prompt_hash.
    Raises KeyError if version not found.
    """
    if version not in PROMPT_REGISTRY:
        raise KeyError(f"Prompt version '{version}' not found. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[version]


def get_active_prompt() -> dict:
    """
    Get the currently active prompt version.
    Returns dict with version, template, prompt_hash, description.
    """
    entry = PROMPT_REGISTRY[PROMPT_VERSION]
    return {
        "version": PROMPT_VERSION,
        "template": entry["template"],
        "prompt_hash": entry["prompt_hash"],
        "description": entry["description"],
    }


def list_versions() -> list[dict]:
    """List all available prompt versions with metadata (no templates)."""
    return [
        {
            "version": v,
            "prompt_hash": e["prompt_hash"],
            "description": e["description"],
            "created_at": e["created_at"],
        }
        for v, e in PROMPT_REGISTRY.items()
    ]
