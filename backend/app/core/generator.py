"""
LLM generation with grounded prompting.

V3 final polish:
  - Retry with exponential backoff (LLM_MAX_RETRIES, default 2)
  - Max concurrency control via Semaphore (LLM_MAX_CONCURRENCY, default 4)
  - Timeout on LLM calls (LLM_TIMEOUT_SECONDS, default 120s)
  - Circuit breaker: after N consecutive failures, skip LLM for cooldown period
  - Supports Ollama (local), Groq/Llama3, Gemini, and OpenAI

Provider priority: Ollama > Groq > Gemini > OpenAI > graceful fallback
"""
import os
import time
import threading
import openai
from app.config import (
    OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT,
    LLM_TIMEOUT_SECONDS, LLM_CIRCUIT_BREAKER_THRESHOLD, LLM_CIRCUIT_BREAKER_COOLDOWN,
    LLM_MAX_RETRIES, LLM_MAX_CONCURRENCY, GROQ_API_KEY, GROQ_MODEL,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


# --- Circuit Breaker ---
class _CircuitBreaker:
    """
    Simple circuit breaker for LLM calls.
    Opens after N consecutive failures, re-closes after cooldown.
    """

    def __init__(self, threshold: int, cooldown_seconds: int):
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (calls should be blocked)."""
        with self._lock:
            if self._failures < self._threshold:
                return False
            # Check if cooldown has elapsed
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._cooldown:
                # Reset: allow a retry
                self._failures = 0
                return False
            return True

    def record_success(self):
        """Reset failure counter on success."""
        with self._lock:
            self._failures = 0

    def record_failure(self):
        """Increment failure counter."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

    @property
    def state(self) -> dict:
        """Current breaker state for observability."""
        with self._lock:
            return {
                "failures": self._failures,
                "threshold": self._threshold,
                "is_open": self._failures >= self._threshold,
                "cooldown_remaining": max(0, self._cooldown - (time.time() - self._last_failure_time))
                if self._failures >= self._threshold else 0,
            }


_circuit = _CircuitBreaker(
    threshold=LLM_CIRCUIT_BREAKER_THRESHOLD,
    cooldown_seconds=LLM_CIRCUIT_BREAKER_COOLDOWN,
)

# --- Concurrency limiter ---
_concurrency_semaphore = threading.Semaphore(LLM_MAX_CONCURRENCY)


def _generate_with_retry(fn, user_prompt: str) -> dict:
    """
    Call an LLM generation function with exponential backoff retry.
    Respects LLM_MAX_RETRIES (default 2 = up to 3 total attempts).
    """
    last_error = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return fn(user_prompt)
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES:
                wait = min(2 ** attempt, 8)  # 1s, 2s, 4s, 8s cap
                time.sleep(wait)
    raise last_error


def _build_context_block(chunks: list[dict]) -> str:
    """
    Format reranked chunks into a labeled context block for the LLM.
    Each chunk is tagged with its source and page for citation.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page", "N/A")
        text = chunk.get("document", "")
        context_parts.append(
            f"[Context {i}] (Source: {source}, Page: {page})\n{text}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assemble the full user prompt with context and question.
    Returns the user message content (system prompt is separate).
    """
    context_block = _build_context_block(chunks)

    user_prompt = f"""Based on the following context, answer the question.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER:"""
    return user_prompt


def _generate_openai(user_prompt: str) -> dict:
    """Generate using OpenAI API."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model": LLM_MODEL,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def _generate_gemini(user_prompt: str) -> dict:
    """Generate using Google Gemini via OpenAI-compatible endpoint."""
    client = openai.OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model": "gemini-2.0-flash",
        "usage": {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) or 0,
            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) or 0,
            "total_tokens": getattr(response.usage, 'total_tokens', 0) or 0,
        },
    }


def _generate_groq(user_prompt: str) -> dict:
    """Generate using Groq API (Llama3 — free, fast, works on HF Spaces)."""
    client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model": f"groq/{GROQ_MODEL}",
        "usage": {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) or 0,
            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) or 0,
            "total_tokens": getattr(response.usage, 'total_tokens', 0) or 0,
        },
    }


def _check_ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        client = openai.OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        client.models.list()
        return True
    except Exception:
        return False


def _generate_ollama(user_prompt: str) -> dict:
    """Generate using local Ollama instance (OpenAI-compatible API)."""
    client = openai.OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "model": f"ollama/{OLLAMA_MODEL}",
        "usage": {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) or 0,
            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) or 0,
            "total_tokens": getattr(response.usage, 'total_tokens', 0) or 0,
        },
    }


def get_active_provider() -> str:
    """Return which LLM provider is configured. Ollama > Groq > Gemini > OpenAI."""
    if _check_ollama_available():
        return "ollama"
    if GROQ_API_KEY:
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


def get_circuit_breaker_state() -> dict:
    """Public accessor for circuit breaker state (observability)."""
    return _circuit.state


def generate(query: str, reranked_chunks: list[dict]) -> dict:
    """
    Generate a grounded answer using the best available LLM.
    Priority: Ollama (local/free) > Groq (Llama3/free) > Gemini > OpenAI > graceful fallback.

    Includes:
      - Timeout enforcement (LLM_TIMEOUT_SECONDS)
      - Circuit breaker (opens after N consecutive failures)

    Returns:
        dict with keys: answer, final_prompt, model, usage
    """
    user_prompt = build_prompt(query, reranked_chunks)
    full_prompt = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}"

    provider = get_active_provider()

    if provider == "none":
        chunk_summary = "\n".join(
            f"- [{c.get('metadata', {}).get('source', '?')}, Page {c.get('metadata', {}).get('page', '?')}]: "
            f"{c.get('document', '')[:150]}..."
            for c in reranked_chunks
        )
        return {
            "answer": (
                "⚠️ No LLM available. Options:\n"
                "1. Install Ollama (free, local): ollama pull qwen2.5:3b\n"
                "2. Set GROQ_API_KEY in HF Space secrets (free, Llama3)\n"
                "3. Set GEMINI_API_KEY in HF Space secrets (free)\n"
                "4. Set OPENAI_API_KEY in HF Space secrets (paid)\n\n"
                "Retrieval and reranking succeeded. Top chunks:\n\n"
                f"{chunk_summary}"
            ),
            "final_prompt": full_prompt,
            "model": "none",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Circuit breaker check
    if _circuit.is_open:
        breaker_state = _circuit.state
        return {
            "answer": (
                f"⚠️ LLM circuit breaker OPEN ({breaker_state['failures']} consecutive failures). "
                f"Cooldown: {breaker_state['cooldown_remaining']:.0f}s remaining.\n\n"
                "Retrieval and reranking succeeded. Check the Sources and Debug panels "
                "below to see the retrieved chunks and scores."
            ),
            "final_prompt": full_prompt,
            "model": f"{provider}/circuit_open",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Acquire concurrency slot
    _concurrency_semaphore.acquire()
    try:
        if provider == "ollama":
            result = _generate_with_retry(_generate_ollama, user_prompt)
        elif provider == "groq":
            result = _generate_with_retry(_generate_groq, user_prompt)
        elif provider == "gemini":
            result = _generate_with_retry(_generate_gemini, user_prompt)
        else:
            result = _generate_with_retry(_generate_openai, user_prompt)

        result["final_prompt"] = full_prompt
        _circuit.record_success()
        return result

    except Exception as e:
        _circuit.record_failure()
        breaker_state = _circuit.state
        return {
            "answer": (
                f"⚠️ LLM generation failed ({provider}) after {LLM_MAX_RETRIES + 1} attempts: {str(e)}\n"
                f"Circuit breaker: {breaker_state['failures']}/{breaker_state['threshold']} failures.\n\n"
                "Retrieval and reranking succeeded. Check the Sources and Debug panels "
                "below to see the retrieved chunks and scores."
            ),
            "final_prompt": full_prompt,
            "model": provider,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    finally:
        _concurrency_semaphore.release()