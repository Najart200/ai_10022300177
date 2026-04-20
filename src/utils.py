"""
utils.py — Small shared utilities for logging setup and UI helpers.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to output to stdout with a clean format."""
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def truncate(text: str, max_len: int = 300, suffix: str = "…") -> str:
    """Truncate text to max_len characters for UI display."""
    return text if len(text) <= max_len else text[:max_len] + suffix


def score_colour(score: float) -> str:
    """Map a [0,1] similarity score to a hex colour (green→yellow→red)."""
    if score >= 0.75:
        return "#22c55e"    # green
    elif score >= 0.50:
        return "#f59e0b"    # amber
    else:
        return "#ef4444"    # red
