"""
Pluggable LLM backend so the generator (and evaluator) can run against
whatever model you have available, with zero required paid API key.

Backends:
  - "ollama"    (default): local model via Ollama's HTTP API. Free, runs
                 entirely on your machine. Install: https://ollama.com,
                 then `ollama pull llama3.2`.
  - "openai":    set LLM_PROVIDER=openai and OPENAI_API_KEY, OPENAI_MODEL
                 (default gpt-4o-mini).
  - "anthropic": set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY,
                 ANTHROPIC_MODEL (default claude-3-5-haiku-latest).
  - "mock":      no network/model at all - returns a deterministic
                 templated response. NOT for grading; only used to smoke
                 test that the retrieval + prompting + I/O pipeline runs
                 end-to-end in environments with no model access.

Select the backend with the LLM_PROVIDER env var (default: ollama).
"""

import os
import json
import urllib.request
import urllib.error


PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")


class LLMError(RuntimeError):
    pass


def _post_json(url, payload, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LLMError("Request to %s failed: %s" % (url, e)) from e


def _call_ollama(system_prompt, user_prompt, temperature=0.4):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    result = _post_json(OLLAMA_HOST + "/api/chat", payload)
    return result["message"]["content"]


def _call_openai(system_prompt, user_prompt, temperature=0.4):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY not set")
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    result = _post_json(
        "https://api.openai.com/v1/chat/completions",
        payload,
        headers={"Authorization": "Bearer " + key},
    )
    return result["choices"][0]["message"]["content"]


def _call_anthropic(system_prompt, user_prompt, temperature=0.4):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    result = _post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return "".join(block.get("text", "") for block in result.get("content", []))


def _call_mock(system_prompt, user_prompt, temperature=0.4):
    # Deterministic, offline stand-in so the pipeline is testable without
    # any model installed. NOT for grading - see module docstring.
    if "Respond with ONLY valid JSON" in system_prompt:
        return _mock_judge_response(user_prompt)
    marker = "Now write a reply to this new customer email"
    tail = user_prompt.split(marker)[-1].strip()
    return (
        "Hi there,\n\nThanks for reaching out. [MOCK BACKEND - no real model was "
        "called; set LLM_PROVIDER=ollama/openai/anthropic for a real generated reply.] "
        "Here is a summary of what you told us:\n" + tail[:200] + "\n\n"
        "We're looking into this and will follow up shortly with next steps.\n\nBest,\nSupport Team"
    )


def _mock_judge_response(user_prompt):
    # Very rough heuristic scorer so evaluate.py is testable offline:
    # penalizes obviously bad signals (mock marker, very short replies)
    # and otherwise returns a plausible mid-high score. This is only a
    # smoke test of the pipeline, not a real judge - real scoring
    # requires a real LLM_PROVIDER.
    is_mock_reply = "[MOCK BACKEND" in user_prompt
    base = 2 if is_mock_reply else 4
    dims = ["relevance", "completeness", "correctness", "tone", "actionability", "conciseness"]
    obj = {d: {"score": base, "why": "mock judge heuristic score"} for d in dims}
    return json.dumps(obj)


_BACKENDS = {
    "ollama": _call_ollama,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "mock": _call_mock,
}


def chat(system_prompt, user_prompt, temperature=0.4, provider=None):
    """Call the configured LLM backend and return the text response."""
    provider = (provider or PROVIDER).lower()
    if provider not in _BACKENDS:
        raise LLMError("Unknown LLM_PROVIDER '%s'. Choose from %s" % (provider, list(_BACKENDS)))
    return _BACKENDS[provider](system_prompt, user_prompt, temperature)
