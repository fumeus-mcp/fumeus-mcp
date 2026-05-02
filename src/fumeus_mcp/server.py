"""MCP server exposing the Fumeus smoke-term analysis workflows."""

from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import adapters

mcp = FastMCP(
    "Fumeus",
    instructions=(
        "Analyze text with Fumeus smoke-term workflows. "
        "Use generate_smoke_terms for labeled CSV datasets and score_records for scoring "
        "records with a weighted term dictionary."
    ),
    host=os.getenv("FUMEUS_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("FUMEUS_MCP_PORT", "8000")),
    json_response=True,
    stateless_http=True,
)


@mcp.tool()
def upload_file(
    file_name: str,
    content: str,
    encoding: str = "text",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Upload a file for later use by Fumeus tools and return its server-side path."""

    return adapters.upload_file(
        file_name=file_name,
        content=content,
        encoding=encoding,
        overwrite=overwrite,
    )


@mcp.tool()
def generate_smoke_terms(
    dataset_path: str,
    header: bool = False,
    text_column: int = 0,
    label_column: int = -1,
    ngram_length: int = 1,
    mode: str = "CC",
    top_n: int = 200,
    output_format: str = "json",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate ranked smoke terms from a labeled CSV dataset."""

    return adapters.generate_smoke_terms(
        dataset_path=dataset_path,
        header=header,
        text_column=text_column,
        label_column=label_column,
        ngram_length=ngram_length,
        mode=mode,
        top_n=top_n,
        output_format=output_format,
        output_path=output_path,
    )


@mcp.tool()
def score_records(
    dataset_path: str,
    terms_path: str,
    header: bool = False,
    terms_header: bool = False,
    text_column: int = 0,
    output_format: str = "json",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Score CSV records with a weighted smoke-term dictionary."""

    return adapters.score_records(
        dataset_path=dataset_path,
        terms_path=terms_path,
        header=header,
        terms_header=terms_header,
        text_column=text_column,
        output_format=output_format,
        output_path=output_path,
    )


@mcp.tool()
def clean_text(texts: list[str], stop_words: list[str] | None = None) -> dict[str, Any]:
    """Return Fumeus-cleaned token lists for supplied text strings."""

    return adapters.clean_text(texts=texts, stop_words=stop_words)


@mcp.tool()
def score_text(text: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one text string with in-memory term weights."""

    return adapters.score_text(text=text, terms=terms)


def main() -> None:
    """Run the Fumeus MCP server."""

    parser = argparse.ArgumentParser(description="Run the Fumeus MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=os.getenv("FUMEUS_MCP_TRANSPORT", "stdio"),
        help="MCP transport to use. Defaults to stdio.",
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
