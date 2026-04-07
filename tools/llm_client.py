"""
Unified LLM client for Polaris Ahmadi.
Supports Gemini (Google) and Qwen 2.5 (Alibaba DashScope via OpenAI-compatible API).
"""

import logging
import os
from typing import Optional

_logger = logging.getLogger(__name__)


def generate_text(
    prompt: str,
    api_key: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    qwen_base_url: Optional[str] = None,
) -> str:
    """
    Generate text from an LLM. Routes to Gemini or Qwen based on provider.

    Args:
        prompt: The text prompt to send.
        api_key: API key (Gemini key for provider=gemini, DashScope key for provider=qwen).
        provider: "gemini" or "qwen".
        model: Model ID. Defaults: gemini-2.5-flash-lite, qwen2.5-72b-instruct.
        qwen_base_url: Base URL for Qwen (DashScope). Default: Beijing region.

    Returns:
        Generated text from the model.
    """
    provider = (provider or "gemini").lower().strip()
    if provider not in ("gemini", "qwen"):
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'qwen'.")

    if not api_key or not api_key.strip():
        key_name = "Gemini" if provider == "gemini" else "DashScope (Qwen)"
        raise ValueError(f"API key is empty. Please set your {key_name} API key in Settings.")

    if provider == "gemini":
        return _generate_gemini(prompt, api_key, model or "gemini-2.5-flash-lite")
    else:
        return _generate_qwen(
            prompt,
            api_key,
            model or "qwen2.5-72b-instruct",
            qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def _generate_gemini(prompt: str, api_key: str, model_id: str) -> str:
    """Generate text using Google Gemini."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import google.generativeai as genai  # type: ignore
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(prompt)
    if not response or not (hasattr(response, "text") and response.text):
        raise ValueError("No text in Gemini response")
    return response.text


def _generate_qwen(prompt: str, api_key: str, model_id: str, base_url: str) -> str:
    """Generate text using Qwen via DashScope OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for Qwen. Install with: pip install openai"
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response or not response.choices:
        raise ValueError("No response from Qwen API")
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty content in Qwen response")
    return content
