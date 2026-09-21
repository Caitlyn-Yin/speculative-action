import openai

from . import constants


class LLMClient:
    """LLM client.

    Two backends, selected by ``backend`` (default: ``constants.llm_backend``):

    * ``"local"``  -- an OpenAI-compatible server (vLLM). This is the only
      backend permitted for research runs: no external LLM APIs.
    * ``"external"`` -- the original Gemini / OpenAI / OpenRouter path, kept
      intact behind the flag so historical behaviour stays reachable.

    The external SDKs are imported lazily so the module loads (and the local
    backend works) on a machine where ``google-genai`` is not installed.
    """

    def __init__(self, model_name, temperature, max_tokens, top_p,
                 backend=None, base_url=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.backend = backend if backend is not None else constants.llm_backend

        if self.backend == "local":
            self.base_url = base_url if base_url is not None else constants.actor_base_url
            self.local_client = openai.OpenAI(
                base_url=self.base_url,
                api_key=constants.local_api_key,
                timeout=constants.local_request_timeout,
                max_retries=constants.local_max_retries,
            )
        elif self.backend == "external":
            self.base_url = base_url
            self._gemini_client = None
            self._openai_client = None
            self._openrouter_client = None
        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}; expected 'local' or 'external'"
            )

    # ------------------------------------------------------------------ local

    def _local_call(self, prompt, stop):
        """Greedy completion against a local vLLM server.

        ``enable_thinking=False`` is forwarded to the chat template so Qwen3
        does not emit <think> blocks; the reply is additionally scrubbed by
        :meth:`_strip_thinking` as a belt-and-braces guard.
        """
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        if constants.local_seed is not None:
            extra_body["seed"] = constants.local_seed

        response = self.local_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        return self._strip_thinking(content)

    @staticmethod
    def _strip_thinking(text):
        """Remove any <think>...</think> residue, including an unclosed block."""
        if not text:
            return text
        while "<think>" in text:
            head, _, rest = text.partition("<think>")
            if "</think>" in rest:
                _, _, tail = rest.partition("</think>")
                text = head + tail
            else:
                text = head
                break
        return text.strip()

    # --------------------------------------------------------------- external

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=constants.gemini_api_key)
        return self._gemini_client

    def _get_openai_client(self):
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=constants.openai_api_key)
        return self._openai_client

    def _get_openrouter_client(self):
        if self._openrouter_client is None:
            self._openrouter_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=constants.openrouter_api_key,
            )
        return self._openrouter_client

    # ------------------------------------------------------------- dispatch

    def call(self, prompt, stop=None):
        if self.backend == "local":
            return self._local_call(prompt, stop)
        if self.model_name.startswith("gemini"):
            return self._gemini_call(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self._openai_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _gemini_call(self, prompt, stop):
        from google.genai import types
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        response = self._get_gemini_client().models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return str(response.text)

    def _openai_call(self, prompt, stop):
        response = self._get_openai_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _openrouter_call(self, prompt, stop):
        response = self._get_openrouter_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content
