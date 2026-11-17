"""
LLM client: Google Gemini (cloud API, google-genai). The only LLM used by the app.

    llm.ask(prompt, as_json, model)        one prompt -> text (used by every pipeline)
    llm.generate(contents, config, model)  raw Gemini call with retries (Agentic RAG tools)
    llm.gemini_models()                    models for the dropdown in the UI
"""

import time

from core.config import env, GEMINI_MODEL, LLM_TIMEOUT_MS


class LLMClient:
    def __init__(self):
        self.client = None                      # created by connect()
        self._models_cache = None

    def connect(self):
        """Create the Gemini client (timeout so a stuck request cannot hang a pipeline forever)."""
        key = env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing in .env - the app uses Google Gemini")
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS))
        return self.client

    def gemini_models(self):
        """Gemini text models this API key can use (cached for 10 minutes). Image, audio, TTS,
        live and embedding variants are left out."""
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        if self._models_cache and time.time() - self._models_cache[0] < 600:
            return self._models_cache[1]
        skip = ("image", "tts", "audio", "live", "embed", "vision", "aqa", "robotics", "computer-use",
                "banana", "transcribe")
        models = []
        for m in self.client.models.list():
            name = m.name.split("/")[-1]
            if ("generateContent" in (m.supported_actions or []) and name.startswith("gemini")
                    and not any(s in name for s in skip)):
                models.append({"name": name, "display_name": m.display_name or name})
        models.sort(key=lambda m: m["name"], reverse=True)          # newest versions first
        self._models_cache = (time.time(), models)
        return models

    def generate(self, contents, config, retries=4, model=None):
        """Gemini call with retry + backoff for temporary errors (429 rate limit, 5xx overload)."""
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        for attempt in range(retries + 1):
            try:
                return self.client.models.generate_content(model=model or GEMINI_MODEL,
                                                           contents=contents, config=config)
            except Exception as e:
                daily_quota = "PerDay" in str(e)          # waiting a few seconds will not help
                temporary = ((getattr(e, "code", None) in (429, 500, 503, 504) and not daily_quota)
                             or "timeout" in type(e).__name__.lower())
                if attempt == retries or not temporary:
                    raise
                time.sleep(2 * 2 ** attempt)   # 2s, 4s, 8s, 16s

    def ask(self, prompt, as_json=False, model=None):
        """One Gemini call. model = the one picked in the UI (default LIVE_LLM_MODEL from .env)."""
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json" if as_json else None)
        return self.generate(prompt, config, model=model).text or ""


llm = LLMClient()
