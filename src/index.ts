#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { exec } from 'child_process';
import { promisify } from 'util';
import axios from 'axios';

const execPromise = promisify(exec);

// Get API keys from environment variables if needed
const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;

class UnslothServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'unsloth-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // Set up tool handlers
    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private async checkUnslothInstallation(): Promise<boolean> {
    try {
      await execPromise('python -c "import unsloth"');
      return true;
    } catch (error) {
      return false;
    }
  }

  private async executeUnslothScript(script: string): Promise<string> {
    try {
      const { stdout, stderr } = await execPromise(`python -c "${script}"`);
      if (stderr && !stdout) {
        throw new Error(stderr);
      }
      return stdout;
    } catch (error: any) {
      throw new Error(`Error executing Unsloth script: ${error.message}`);
    }
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'check_installation',
          description: 'Check if Unsloth is properly installed',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'list_supported_models',
          description: 'List all models supported by Unsloth',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'load_model',
          description: 'Load a pretrained model with Unsloth optimizations',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name of the model to load (e.g., "unsloth/Llama-3.2-1B")',
              },
              max_seq_length: {
                type: 'number',
                description: 'Maximum sequence length for the model',
              },
              load_in_4bit: {
                type: 'boolean',
                description: 'Whether to load the model in 4-bit quantization',
              },
              use_gradient_checkpointing: {
                type: 'boolean',
                description: 'Whether to use gradient checkpointing to save memory',
              },
            },
            required: ['model_name'],
          },
        },
        {
          name: 'finetune_model',
          description: 'Fine-tune a model with Unsloth optimizations',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name of the model to fine-tune',
              },
              dataset_name: {
                type: 'string',
                description: 'Name of the dataset to use for fine-tuning',
              },
              output_dir: {
                type: 'string',
                description: 'Directory to save the fine-tuned model',
              },
              max_seq_length: {
                type: 'number',
                description: 'Maximum sequence length for training',
              },
              lora_rank: {
                type: 'number',
                description: 'Rank for LoRA fine-tuning',
              },
              lora_alpha: {
                type: 'number',
                description: 'Alpha for LoRA fine-tuning',
              },
              batch_size: {
                type: 'number',
                description: 'Batch size for training',
              },
              gradient_accumulation_steps: {
                type: 'number',
                description: 'Number of gradient accumulation steps',
              },
              learning_rate: {
                type: 'number',
                description: 'Learning rate for training',
              },
              max_steps: {
                type: 'number',
                description: 'Maximum number of training steps',
              },
              dataset_text_field: {
                type: 'string',
                description: 'Field in the dataset containing the text',
              },
              load_in_4bit: {
                type: 'boolean',
                description: 'Whether to use 4-bit quantization',
              },
            },
            required: ['model_name', 'dataset_name', 'output_dir'],
          },
        },
        {
          name: 'generate_text',
          description: 'Generate text using a fine-tuned Unsloth model',
          inputSchema: {
            type: 'object',
            properties: {
              model_path: {
                type: 'string',
                description: 'Path to the fine-tuned model',
              },
              prompt: {
                type: 'string',
                description: 'Prompt for text generation',
              },
              max_new_tokens: {
                type: 'number',
                description: 'Maximum number of tokens to generate',
              },
              temperature: {
                type: 'number',
                description: 'Temperature for text generation',
              },
              top_p: {
                type: 'number',
                description: 'Top-p for text generation',
              },
            },
            required: ['model_path', 'prompt'],
          },
        },
        {
          name: 'export_model',
          description: 'Export a fine-tuned Unsloth model to various formats',
          inputSchema: {
            type: 'object',
            properties: {
              model_path: {
                type: 'string',
                description: 'Path to the fine-tuned model',
              },
              export_format: {
                type: 'string',
                description: 'Format to export to (gguf, ollama, vllm, huggingface)',
                enum: ['gguf', 'ollama', 'vllm', 'huggingface'],
              },
              output_path: {
                type: 'string',
                description: 'Path to save the exported model',
              },
              quantization_bits: {
                type: 'number',
                description: 'Bits for quantization (for GGUF export)',
              },
            },
            required: ['model_path', 'export_format', 'output_path'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'check_installation': {
            const isInstalled = await this.checkUnslothInstallation();
            
            if (!isInstalled) {
              return {
                content: [
                  {
                    type: 'text',
                    text: 'Unsloth is not installed. Please install it with: pip install unsloth',
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: 'Unsloth is properly installed.',
                },
              ],
            };
          }

          case 'list_supported_models': {
            const script = `
import json
try:
    from unsloth import FastLanguageModel
    # Define a list of supported models
    models = [
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-1B-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-8B-bnb-4bit",
        "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
        "unsloth/Qwen-2.5-7B"
    ]
    print(json.dumps(models))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const models = JSON.parse(result);
              if (models.error) {
                throw new Error(models.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: JSON.stringify(models, null, 2),
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error parsing model list: ${error.message}`);
            }
          }

          case 'load_model': {
            const { model_name, max_seq_length = 2048, load_in_4bit = true, use_gradient_checkpointing = true } = args as {
              model_name: string;
              max_seq_length?: number;
              load_in_4bit?: boolean;
              use_gradient_checkpointing?: boolean;
            };

            const script = `
import json
try:
    from unsloth import FastLanguageModel
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing=${use_gradient_checkpointing ? '"unsloth"' : 'False'}
    )
    
    # Get model info
    model_info = {
        "model_name": "${model_name}",
        "max_seq_length": ${max_seq_length},
        "load_in_4bit": ${load_in_4bit},
        "use_gradient_checkpointing": ${use_gradient_checkpointing},
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type,
        "success": True
    }
    
    print(json.dumps(model_info))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const modelInfo = JSON.parse(result);
              if (!modelInfo.success) {
                throw new Error(modelInfo.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully loaded model: ${model_name}\n\n${JSON.stringify(modelInfo, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error loading model: ${error.message}`);
            }
          }

          case 'finetune_model': {
            const {
              model_name,
              dataset_name,
              output_dir,
              max_seq_length = 2048,
              lora_rank = 16,
              lora_alpha = 16,
              batch_size = 2,
              gradient_accumulation_steps = 4,
              learning_rate = 2e-4,
              max_steps = 100,
              dataset_text_field = 'text',
              load_in_4bit = true,
            } = args as {
              model_name: string;
              dataset_name: string;
              output_dir: string;
              max_seq_length?: number;
              lora_rank?: number;
              lora_alpha?: number;
              batch_size?: number;
              gradient_accumulation_steps?: number;
              learning_rate?: number;
              max_steps?: number;
              dataset_text_field?: string;
              load_in_4bit?: boolean;
            };

            const script = `
import json
import os
try:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    
    # Create output directory if it doesn't exist
    os.makedirs("${output_dir}", exist_ok=True)
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing="unsloth"
    )
    
    # Load the dataset
    dataset = load_dataset("${dataset_name}")
    
    # Patch the model with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=${lora_rank},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=${lora_alpha},
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=${max_seq_length},
        use_rslora=False,
        loftq_config=None
    )
    
    # Configure the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="${dataset_text_field}",
            max_seq_length=${max_seq_length},
            per_device_train_batch_size=${batch_size},
            gradient_accumulation_steps=${gradient_accumulation_steps},
            warmup_steps=10,
            max_steps=${max_steps},
            learning_rate=${learning_rate},
            logging_steps=1,
            output_dir="${output_dir}",
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    print(json.dumps({
        "success": True,
        "output_dir": "${output_dir}",
        "model_name": "${model_name}",
        "dataset_name": "${dataset_name}",
        "max_steps": ${max_steps}
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const trainingResult = JSON.parse(result);
              if (!trainingResult.success) {
                throw new Error(trainingResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully fine-tuned model: ${model_name} with dataset: ${dataset_name}\n\n${JSON.stringify(trainingResult, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error fine-tuning model: ${error.message}`);
            }
          }

          case 'generate_text': {
            const {
              model_path,
              prompt,
              max_new_tokens = 256,
              temperature = 0.7,
              top_p = 0.9,
            } = args as {
              model_path: string;
              prompt: string;
              max_new_tokens?: number;
              temperature?: number;
              top_p?: number;
            };

            const script = `
import json
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=${max_new_tokens},
        temperature=${temperature},
        top_p=${top_p},
        do_sample=True
    )
    
    # Generate text
    result = generator("${prompt.replace(/"/g, '\\"')}")
    
    print(json.dumps({
        "success": True,
        "prompt": "${prompt.replace(/"/g, '\\"')}",
        "generated_text": result[0]["generated_text"]
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const generationResult = JSON.parse(result);
              if (!generationResult.success) {
                throw new Error(generationResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Generated text:\n\n${generationResult.generated_text}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error generating text: ${error.message}`);
            }
          }

          case 'export_model': {
            const {
              model_path,
              export_format,
              output_path,
              quantization_bits = 4,
            } = args as {
              model_path: string;
              export_format: 'gguf' | 'ollama' | 'vllm' | 'huggingface';
              output_path: string;
              quantization_bits?: number;
            };

            let script = '';
            
            if (export_format === 'gguf') {
              script = `
import json
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname("${output_path}"), exist_ok=True)
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Save the model in GGUF format
    from transformers import LlamaForCausalLM
    import ctranslate2
    
    # Convert to GGUF format
    ct_model = ctranslate2.converters.TransformersConverter(
        "${model_path}",
        "${output_path}",
        quantization="int${quantization_bits}"
    ).convert()
    
    print(json.dumps({
        "success": True,
        "model_path": "${model_path}",
        "export_format": "gguf",
        "output_path": "${output_path}",
        "quantization_bits": ${quantization_bits}
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            } else if (export_format === 'huggingface') {
              script = `
import json
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create output directory if it doesn't exist
    os.makedirs("${output_path}", exist_ok=True)
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Save the model in Hugging Face format
    model.save_pretrained("${output_path}")
    tokenizer.save_pretrained("${output_path}")
    
    print(json.dumps({
        "success": True,
        "model_path": "${model_path}",
        "export_format": "huggingface",
        "output_path": "${output_path}"
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            } else {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Export format '${export_format}' is not yet implemented. Currently, only 'gguf' and 'huggingface' formats are supported.`,
                  },
                ],
                isError: true,
              };
            }
            
            const result = await this.executeUnslothScript(script);
            
            try {
              const exportResult = JSON.parse(result);
              if (!exportResult.success) {
                throw new Error(exportResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully exported model to ${export_format} format:\n\n${JSON.stringify(exportResult, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error exporting model: ${error.message}`);
            }
          }

          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error: any) {
        console.error(`Error executing tool ${name}:`, error);
        
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message || 'Unknown error'}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Unsloth MCP server running on stdio');
  }
}

const server = new UnslothServer();
server.run().catch(console.error);
