"""
llm_client.py — Calls an LLM API via raw HTTP requests (no SDK wrapper).
Uses the Groq API (free tier) with llama-3.1-8b-instant by default.

Groq was chosen because:
  - Free tier: 14 400 requests/day, 6 000 tokens/min
  - Fast inference (tokens per second >> OpenAI free tier)
  - OpenAI-compatible endpoint — easy to swap for other providers
  - No need for a paid API key for assignment submission

To get a free key: https://console.groq.com

Alternative: set PROVIDER=huggingface and supply a HF_TOKEN to use
Mistral-7B-Instruct via the HuggingFace Inference API (also free).

Student: Najart Rauf Awuni | Index: 10022300177
"""

import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.1-8b-instant"
HF_API_URL    = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"


def _get_secret(name: str) -> str:
    """
    Read a secret from env vars first, then Streamlit secrets when available.
    This keeps CLI/script usage working while supporting Streamlit deployments.
    """
    value = os.getenv(name, "")
    if value:
        return value

    try:
        import streamlit as st  # optional dependency at runtime
        return st.secrets.get(name, "")
    except Exception:
        return ""


def call_llm(prompt: str,
             temperature: float = 0.2,
             max_tokens: int = 512) -> dict:
    """
    Send prompt to LLM and return a result dict:
    {
      'response':   str  — model's answer text,
      'model':      str  — model ID used,
      'tokens_in':  int  — prompt tokens,
      'tokens_out': int  — completion tokens,
      'provider':   str  — 'groq' | 'huggingface',
      'error':      str | None
    }
    """
    api_key  = _get_secret("GROQ_API_KEY")
    hf_token = _get_secret("HF_TOKEN")

    if api_key:
        return _call_groq(prompt, api_key, temperature, max_tokens)
    elif hf_token:
        return _call_huggingface(prompt, hf_token, max_tokens)
    else:
        logger.warning("No API key found — returning stub response for demo.")
        return {
            "response":   (
                "[Demo mode — no API key set]\n"
                "Please set GROQ_API_KEY or HF_TOKEN in your environment variables "
                "or Streamlit secrets. Get a free Groq key at https://console.groq.com"
            ),
            "model":      "none",
            "tokens_in":  0,
            "tokens_out": 0,
            "provider":   "none",
            "error":      "NO_API_KEY",
        }


def _call_groq(prompt: str, api_key: str,
               temperature: float, max_tokens: int) -> dict:
    """Call Groq OpenAI-compatible chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers,
                             data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data      = resp.json()
        answer    = data["choices"][0]["message"]["content"].strip()
        usage     = data.get("usage", {})
        return {
            "response":   answer,
            "model":      data.get("model", GROQ_MODEL),
            "tokens_in":  usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "provider":   "groq",
            "error":      None,
        }
    except requests.HTTPError as e:
        logger.error("Groq HTTP error: %s", e)
        return _error_result(str(e), "groq")
    except Exception as e:
        logger.error("Groq call failed: %s", e)
        return _error_result(str(e), "groq")


def _call_huggingface(prompt: str, hf_token: str, max_tokens: int) -> dict:
    """Call HuggingFace Inference API (fallback)."""
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs":      prompt,
        "parameters":  {"max_new_tokens": max_tokens, "temperature": 0.2},
    }

    try:
        resp = requests.post(HF_API_URL, headers=headers,
                             json=payload, timeout=60)
        resp.raise_for_status()
        data   = resp.json()
        answer = data[0].get("generated_text", "").replace(prompt, "").strip()
        return {
            "response":   answer,
            "model":      "mistralai/Mistral-7B-Instruct-v0.3",
            "tokens_in":  0,
            "tokens_out": 0,
            "provider":   "huggingface",
            "error":      None,
        }
    except Exception as e:
        logger.error("HuggingFace call failed: %s", e)
        return _error_result(str(e), "huggingface")


def _error_result(msg: str, provider: str) -> dict:
    return {
        "response":   f"LLM call failed: {msg}",
        "model":      "error",
        "tokens_in":  0,
        "tokens_out": 0,
        "provider":   provider,
        "error":      msg,
    }
