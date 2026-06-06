# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) that installs
  dependencies and builds (type-checks) the server on Node 18 and 20.
- MCP registry descriptor (`server.json`) for submission to the official
  Model Context Protocol registry.
- "Install / Use with Claude Desktop" section in the README.
- This changelog.

## [1.0.0] - Release-notes draft (not yet tagged)

First public release of the Unsloth MCP Server.

### Added
- Stdio MCP server (`unsloth-server`) built on `@modelcontextprotocol/sdk`.
- Tools:
  - `check_installation` — verify Unsloth is installed in the Python env.
  - `list_supported_models` — list models supported by Unsloth.
  - `load_model` — load a model with Unsloth optimizations (4-bit, gradient
    checkpointing, configurable max sequence length).
  - `finetune_model` — LoRA/QLoRA fine-tuning with configurable rank, alpha,
    batch size, gradient accumulation, learning rate, and steps.
  - `generate_text` — text generation from a fine-tuned model.
  - `export_model` — export to GGUF, Ollama, vLLM, or Hugging Face formats.
- Optional `HUGGINGFACE_TOKEN` environment variable for gated/private models.
- Apache-2.0 license.

[Unreleased]: https://github.com/OtotaO/unsloth-mcp-server/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/OtotaO/unsloth-mcp-server/releases/tag/v1.0.0
