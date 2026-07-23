"""Clean raw fingerspelled text into a grammatical prompt using a local LLM.

Primary path: a local Ollama model (default llama3.1:8b). If Ollama is not
running / not installed, falls back to a trivial cleanup so the app still runs.

Ollama is stdlib-only here (urllib) so no extra pip deps are needed.
Install Ollama from https://ollama.com then:  ollama pull llama3.1:8b
"""
import json
import urllib.request
import urllib.error

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

SYSTEM = (
    "You receive raw text assembled from an ASL fingerspelling recognizer. "
    "It may contain misspellings, missing spaces, or ASL-style grammar "
    "(no articles, different word order, no 'to be'). "
    "Rewrite it as a single clear, grammatical English sentence that reads as "
    "a natural prompt to an AI assistant. Reply with ONLY the corrected "
    "sentence, no explanation, no quotes."
)


def _fallback(raw: str) -> str:
    s = " ".join(raw.split()).strip().lower()
    return s[:1].upper() + s[1:] if s else s


def refine(raw: str, model: str = MODEL, host: str = OLLAMA_HOST,
           timeout: float = 30.0) -> str:
    """Return a cleaned prompt. Never raises — falls back on any error."""
    raw = raw.strip()
    if not raw:
        return ""
    payload = json.dumps({
        "model": model,
        "system": SYSTEM,
        "prompt": raw,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out.get("response", "").strip() or _fallback(raw)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return _fallback(raw)


def ollama_available(host: str = OLLAMA_HOST) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "HELO WRLD I WANT COFFE"
    print("raw   :", text)
    print("ollama:", "up" if ollama_available() else "down (using fallback)")
    print("clean :", refine(text))
