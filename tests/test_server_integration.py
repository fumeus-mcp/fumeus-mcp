from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
EXAMPLE_DATASET = ROOT_DIR / "samples" / "example_dataset.csv"
EXAMPLE_DICTIONARY = ROOT_DIR / "samples" / "example_dictionary.csv"


def test_mcp_server_lists_tools_and_runs_generate_smoke_terms(tmp_path):
    async def run_check():
        env = os.environ.copy()
        pythonpath = str(SRC_DIR)
        if env.get("PYTHONPATH"):
            pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
        env["PYTHONPATH"] = pythonpath
        env["FUMEUS_MCP_UPLOAD_DIR"] = str(tmp_path)

        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "fumeus_mcp.server"],
            env=env,
        )

        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}

                assert tool_names == {
                    "upload_file",
                    "generate_smoke_terms",
                    "score_records",
                    "clean_text",
                    "score_text",
                }

                upload_tool = next(tool for tool in tools_result.tools if tool.name == "upload_file")
                assert upload_tool.inputSchema["required"] == ["file_name", "content"]

                upload_result = await session.call_tool(
                    "upload_file",
                    {
                        "file_name": "datasets/example_dataset.csv",
                        "content": EXAMPLE_DATASET.read_text(encoding="utf-8"),
                    },
                )

                assert upload_result.isError is False
                assert upload_result.structuredContent["path"] == str(
                    tmp_path / "datasets" / "example_dataset.csv"
                )
                assert upload_result.structuredContent["size_bytes"] > 0

                generate_tool = next(
                    tool for tool in tools_result.tools if tool.name == "generate_smoke_terms"
                )
                assert generate_tool.inputSchema["required"] == ["dataset_path"]
                assert "label_column" in generate_tool.inputSchema["properties"]

                result = await session.call_tool(
                    "generate_smoke_terms",
                    {
                        "dataset_path": upload_result.structuredContent["path"],
                        "header": True,
                        "text_column": 0,
                        "label_column": 1,
                        "ngram_length": 1,
                        "mode": "DRC",
                        "top_n": 3,
                        "output_format": "json",
                    },
                )

                assert result.isError is False
                assert result.structuredContent["count"] == 3
                assert result.structuredContent["mode"] == "DRC"
                assert result.structuredContent["ngram_length"] == 1
                assert {term["term"] for term in result.structuredContent["terms"]} == {
                    "staff",
                    "rude",
                    "were",
                }
                assert {
                    term["score"] for term in result.structuredContent["terms"]
                } == {2.82842712474619}

    asyncio.run(run_check())


def test_mcp_server_runs_score_records():
    async def run_check():
        env = os.environ.copy()
        pythonpath = str(SRC_DIR)
        if env.get("PYTHONPATH"):
            pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
        env["PYTHONPATH"] = pythonpath

        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "fumeus_mcp.server"],
            env=env,
        )

        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                score_tool = next(tool for tool in tools_result.tools if tool.name == "score_records")
                assert score_tool.inputSchema["required"] == ["dataset_path", "terms_path"]
                assert "terms_header" in score_tool.inputSchema["properties"]

                result = await session.call_tool(
                    "score_records",
                    {
                        "dataset_path": str(EXAMPLE_DATASET),
                        "terms_path": str(EXAMPLE_DICTIONARY),
                        "header": True,
                        "terms_header": True,
                        "text_column": 0,
                        "output_format": "json",
                    },
                )

                assert result.isError is False
                assert result.structuredContent == {
                    "records": [
                        {
                            "text": "Service was unacceptable. The staff were rude!",
                            "score": 12.618802154,
                            "terms_found": ["rude", "staff", "unacceptable", "service"],
                        },
                        {
                            "text": "Staff were so rude to me.",
                            "score": 8.0,
                            "terms_found": ["rude", "staff"],
                        },
                        {
                            "text": "Great experience!",
                            "score": 0,
                            "terms_found": [],
                        },
                        {
                            "text": "I loved my visit and would definitely return!",
                            "score": 0,
                            "terms_found": [],
                        },
                    ],
                    "count": 4,
                }

    asyncio.run(run_check())
