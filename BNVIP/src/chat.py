"""Conversational agent backed by a local Ollama model, with memory.

The user "types" by signing keyword words; refine_prompt turns those into a
clean sentence, and Chat.send() keeps a running conversation and returns the
agent's reply. Falls back to a canned reply if Ollama is unavailable.
"""
import json
import urllib.request
import urllib.error

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

SYSTEM = (
    "You are a friendly, concise assistant in a live conversation with a Deaf "
    "or hard-of-hearing user who communicates through sign language. Their "
    "messages arrive as short, possibly imperfect English (recognized from "
    "signs). Infer their intent charitably and reply helpfully in 1-2 short "
    "sentences. Ask a brief clarifying question when needed. Keep it natural "
    "and warm."
)


class Chat:
    def __init__(self, model: str = MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
        self.history = []          # list of {role, content}

    def send(self, user_text: str, timeout: float = 30.0) -> str:
        """Add a user turn, get the agent reply, keep the conversation."""
        user_text = user_text.strip()
        if not user_text:
            return ""
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM}] + self.history
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "stream": False, "options": {"temperature": 0.4},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            reply = out.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            reply = "(LLM offline) I heard: " + user_text
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.history = []
