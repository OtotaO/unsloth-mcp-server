# Unsloth MCP Server

An MCP server for [Unsloth](https://github.com/unslothai/unsloth) - a library that makes LLM fine-tuning 2x faster with 80% less memory.

## Features

- Optimize fine-tuning for Llama, Mistral, Phi, Gemma, and other models
- 4-bit quantization for efficient training
- Extended context length support
- Simple API for model loading, fine-tuning, and inference

## Quick Start

1. Install Unsloth: `pip install unsloth`
2. Install and build the server:
   ```bash
   cd unsloth-server
   npm install
   npm run build
   ```
3. Add to MCP settings:
   ```json
   {
     "mcpServers": {
       "unsloth-server": {
         "command": "node",
         "args": ["/path/to/unsloth-server/build/index.js"],
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

## Available Tools

- **check_installation**: Verify Unsloth installation
- **list_supported_models**: List supported models
- **load_model**: Load a model with Unsloth optimizations
- **finetune_model**: Fine-tune with LoRA/QLoRA
- **generate_text**: Generate text with fine-tuned models
- **export_model**: Export to GGUF, Hugging Face, etc.

## Example

```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "finetune_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    dataset_name: "tatsu-lab/alpaca",
    output_dir: "./fine-tuned-model"
  }
});
```

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA support (recommended)
- Node.js and npm

## License

Apache-2.0
