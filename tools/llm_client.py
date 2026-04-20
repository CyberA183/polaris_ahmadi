"""
Unified LLM client for Polaris Ahmadi.
Supports Qwen via Hugging Face OpenAI-compatible router.
"""

import logging
import os
from typing import Optional

_logger = logging.getLogger(__name__)


def generate_text(
    prompt: str,
    api_key: str,
    provider: str = "qwen",
    model: Optional[str] = None,
    qwen_base_url: Optional[str] = None,
) -> str:
    """
    Generate text from Qwen.

    Args:
        prompt: The text prompt to send.
        api_key: Hugging Face API key.
        provider: Must be "qwen".
        model: Model ID. Defaults: Qwen/Qwen2.5-72B-Instruct.
        qwen_base_url: Base URL for Qwen (HF router). Default: Hugging Face router.

    Returns:
        Generated text from the model.
    """
    provider = (provider or "qwen").lower().strip()
    if provider != "qwen":
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'qwen'.")

    if not api_key or not api_key.strip():
        raise ValueError("API key is empty. Please set your Hugging Face API key in Settings.")

    return _generate_qwen(
        prompt,
        api_key,
        model or "Qwen/Qwen2.5-72B-Instruct",
        qwen_base_url or "https://router.huggingface.co/v1",
    )


def _generate_qwen(prompt: str, api_key: str, model_id: str, base_url: str) -> str:
    """Generate text using Qwen via Hugging Face OpenAI-compatible API."""
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
