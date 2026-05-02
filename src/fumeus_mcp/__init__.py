"""MCP wrapper for the Fumeus smoke-term analysis tools."""

from .adapters import (
    clean_text,
    generate_smoke_terms,
    score_records,
    score_text,
    upload_file,
)

__all__ = [
    "clean_text",
    "generate_smoke_terms",
    "score_records",
    "score_text",
    "upload_file",
]
