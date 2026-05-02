# Fumeus

Fumeus is a family of Python tools for smoke-term analysis. Smoke terms are words and phrases that are especially indicative of a target condition in text, such as safety hazard language or customer concern language.

This repository includes the original Fumeus Python functions plus an MCP server wrapper for use from MCP-capable clients.

See the [sample conversation](samples/Sample%20conversation.pdf) for an example of usage.

## What It Does

- `generate_smoke_terms`: generate ranked n-gram smoke terms from a labeled CSV dataset.
- `score_records`: score CSV records with a weighted smoke-term dictionary.
- `clean_text`: return the tokens produced by Fumeus text cleaning.
- `score_text`: score one text string with an in-memory list of term weights.
- `upload_file`: save user-provided text or base64 file contents for later MCP tool calls.

## Samples

- `samples/example_dataset.csv`: sample records with text in `Record` and labels in `Concern?`.
- `samples/example_dictionary.csv`: sample smoke-term dictionary with `Term` and `Weight`.

## Project Layout

```text
.
├── generate.py
├── score.py
├── fum_utils.py
├── pyproject.toml
├── samples/
├── src/fumeus_mcp/
└── tests/
```

## Local Setup

1. Clone the repository and enter it:

```bash
git clone <your-fumeus-main-url>
cd fumeus-main
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install Fumeus and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Runtime and test dependencies are declared in `pyproject.toml`.

4. Run the tests:

```bash
python -m pytest -q
```

## Quick Start: ChatGPT

ChatGPT custom MCP connectors use remote MCP servers. A local `localhost` server is not enough; ChatGPT needs an HTTPS endpoint it can reach from the public internet.

1. Complete the local setup above.

2. Start the MCP server with HTTP transport:

```bash
FUMEUS_MCP_HOST=0.0.0.0 FUMEUS_MCP_PORT=8000 fumeus-mcp --transport streamable-http
```

3. Expose or deploy that server at an HTTPS URL:

```text
https://your-domain.example/mcp
```

4. In ChatGPT web, enable developer mode for custom MCP connectors.

5. Create a new custom MCP connector or app.

6. Set the connector endpoint to your HTTPS MCP URL.

7. For a private test endpoint, choose no authentication. For anything shared or deployed, configure authentication and review the server’s file access.

8. Optional: set `FUMEUS_MCP_UPLOAD_DIR` to choose where uploaded files are stored. By default, uploads are saved under `uploads/` in this project.

9. Save the connector, enable it in a chat, and upload or test a sample file with a prompt such as:

```text
Upload a file named my_dataset.csv with this CSV content, then use the returned path to generate smoke terms: Record,Concern? ...
```

10. Test an existing server-side sample file with a prompt such as:

```text
Use Fumeus to generate smoke terms from samples/example_dataset.csv with text_column 0, label_column 1, ngram_length 1, mode DRC, and top_n 3.
```

11. Test scoring with:

```text
Use Fumeus to score samples/example_dataset.csv with samples/example_dictionary.csv. The dataset and dictionary both have headers, and the text column is 0.
```

## Quick Start: Claude Desktop

Claude Desktop can launch local MCP servers over stdio. This is separate from Claude remote connectors in claude.ai, which require an internet-reachable server.

1. Complete the local setup above.

2. Verify the stdio server command works:

```bash
fumeus-mcp --help
```

3. Open your Claude Desktop config file.

On macOS:

```text
~/Library/Application Support/Claude/claude_desktop_config.json
```

On Windows:

```text
%APPDATA%\Claude\claude_desktop_config.json
```

4. Add a local MCP server entry. Replace `/absolute/path/to/fumeus-main` with this repository’s absolute path:

```json
{
  "mcpServers": {
    "fumeus": {
      "command": "/bin/bash",
      "args": [
        "-lc",
        "cd /absolute/path/to/fumeus-main && .venv/bin/fumeus-mcp"
      ]
    }
  }
}
```

5. Restart Claude Desktop.

6. Start a new Claude chat and test with:

```text
Use Fumeus to generate smoke terms from samples/example_dataset.csv with text_column 0, label_column 1, ngram_length 1, mode DRC, and top_n 3.
```

7. Test scoring with:

```text
Use Fumeus to score samples/example_dataset.csv with samples/example_dictionary.csv. The dataset and dictionary both have headers, and the text column is 0.
```

## Quick Start: Claude Remote Connectors

Claude remote connectors use an internet-reachable MCP server, similar to ChatGPT. Run this server with `--transport streamable-http`, deploy it behind HTTPS, then add the HTTPS `/mcp` endpoint in Claude’s custom connector settings.

## Example Tool Calls

Generate smoke terms from the sample labeled records:

```json
{
  "tool": "generate_smoke_terms",
  "arguments": {
    "dataset_path": "samples/example_dataset.csv",
    "header": true,
    "text_column": 0,
    "label_column": 1,
    "ngram_length": 1,
    "mode": "DRC",
    "top_n": 3,
    "output_format": "json"
  }
}
```

Upload a CSV file over MCP and use the returned `path` as `dataset_path` or `terms_path` in later calls:

```json
{
  "tool": "upload_file",
  "arguments": {
    "file_name": "my_dataset.csv",
    "content": "Record,Concern?\nService was unacceptable,1\nGreat experience,0\n",
    "encoding": "text"
  }
}
```

Score the same records with the sample dictionary:

```json
{
  "tool": "score_records",
  "arguments": {
    "dataset_path": "samples/example_dataset.csv",
    "terms_path": "samples/example_dictionary.csv",
    "header": true,
    "terms_header": true,
    "text_column": 0,
    "output_format": "json"
  }
}
```

Clean one record:

```json
{
  "tool": "clean_text",
  "arguments": {
    "texts": ["Service was unacceptable. The staff were rude!"]
  }
}
```

Score one record in memory:

```json
{
  "tool": "score_text",
  "arguments": {
    "text": "Service was unacceptable. The staff were rude!",
    "terms": [
      { "term": "rude", "weight": 4 },
      { "term": "staff", "weight": 4 },
      { "term": "unacceptable", "weight": 2.309401077 },
      { "term": "service", "weight": 2.309401077 }
    ]
  }
}
```

## Notes

- File paths are resolved from the MCP server process, so run commands from the repository root or pass absolute paths.
- Uploaded files are stored in `FUMEUS_MCP_UPLOAD_DIR` when set, otherwise in `uploads/`. Upload names must be relative paths inside that directory.
- The MCP server can read and write local files accessible to the server process. Only deploy it in environments where that file access is appropriate.
- Custom MCP connectors should only be connected to ChatGPT when you trust the server and the data it can access.
