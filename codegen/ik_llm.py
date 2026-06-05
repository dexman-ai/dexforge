"""LLM invocation module for ik_generator.

This module handles communication with LLM APIs for translating
ssik Python IK artifacts to Rust code.
"""

from __future__ import annotations

import os
from typing import Optional


class IKGeneratorLLM:
    """Thin wrapper around the OpenAI-compatible chat API for IK generation.

    Supports local, air-gapped models (vLLM, Ollama, LM Studio, …) via
    environment variables::

        DEXFORGE_LLM_KEY       API key (default: "nokeyneeded")
        DEXFORGE_LLM_BASE_URL  Base URL (default: https://api.openai.com/v1)
        DEXFORGE_LLM_MODEL     Model name (default: "gpt-4o")

    :param api_key: Override ``DEXFORGE_LLM_KEY``.
    :param base_url: Override ``DEXFORGE_LLM_BASE_URL``.
    :param model: Override ``DEXFORGE_LLM_MODEL``.
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    SYSTEM_PROMPT = (
        "You are an expert Rust systems programmer and robotics engineer. "
        "Translate the provided ssik Python IK artifact to #![no_std] Rust "
        "exactly as instructed. "
        "Also generate the required Python helper scripts. "
        "Output ALL files using the exact '=== FILE: <filename> ===' separator "
        "format described in the prompt. Do not include any conversational text."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLM generation. "
                "Install it with: pip install openai"
            ) from exc

        self._api_key = api_key or os.environ.get("DEXFORGE_LLM_KEY", "nokeyneeded")
        self._base_url = base_url or os.environ.get(
            "DEXFORGE_LLM_BASE_URL", self.DEFAULT_BASE_URL
        )
        self._model = model or os.environ.get("DEXFORGE_LLM_MODEL", self.DEFAULT_MODEL)

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate(self, messages: list[dict]) -> str:
        """Send a conversation *messages* list to the LLM and return the raw response.

        Temperature is fixed at ``0.0`` (greedy decoding) to ensure
        deterministic output across identical inputs.

        :param messages: OpenAI-style chat messages list (system + user turns).
        :returns: The LLM's response as a raw string.
        :raises RuntimeError: If the API call fails or returns an empty choice.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=messages,
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("LLM returned an empty response (no choices).")
        content = choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned a None message content.")
        return content
