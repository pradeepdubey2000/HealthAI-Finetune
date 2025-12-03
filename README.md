# HealthAI-Finetune

A project for fine-tuning Llama 3 language models on medical and healthcare data using QLoRA (Quantized Low-Rank Adaptation) techniques. This repository contains tools for preparing healthcare datasets and fine-tuning large language models for medical chat applications.

## 📋 Project Overview

HealthAI-Finetune enables researchers and practitioners to:
- Prepare healthcare datasets in the proper format for instruction-tuning
- Fine-tune Llama 3 models on medical data using efficient LoRA adapters
- Leverage 4-bit quantization to reduce memory requirements
- Deploy fine-tuned medical chat models with minimal computational resources

## 📁 Project Structure

```
HealthAI-Finetune/
├── dataset-making.ipynb       # Notebook for preparing and formatting healthcare datasets
├── finetune_llama3.ipynb      # Notebook for fine-tuning Llama 3 on medical data
└── README.md                   # This file
```

## 🔧 Notebooks Description

### 1. `dataset-making.ipynb`
Prepares healthcare datasets for fine-tuning by:
- Loading the ChatDoctor-HealthCareMagic-100k dataset from Hugging Face
- Converting datasets to pandas DataFrames for manipulation
- Formatting data into instruction-response pairs with Llama 3 chat templates
- Uploading the formatted dataset to Hugging Face Hub

**Key outputs:**
- Formatted dataset with system prompts, user questions, and assistant responses
- Dataset pushed to Hugging Face Hub for easy access during training

### 2. `finetune_llama3.ipynb`
Fine-tunes Llama 3 models using:
- **Unsloth framework**: For memory-efficient 4x faster training
- **QLoRA (Quantized LoRA)**: Reduces model size with 4-bit quantization
- **Supervised Fine-Tuning (SFT)**: Trains on medical instruction-response pairs
- **LoRA adapters**: Target modules for efficient parameter tuning

**Key features:**
- 4-bit quantization for reduced GPU memory usage
- LoRA configuration with r=16 for low-rank adaptation
- Configurable training parameters (batch size, learning rate, warmup steps)
- Memory statistics tracking
- Model inference testing
- Automatic model upload to Hugging Face Hub

## 📊 Configuration

The fine-tuning notebook uses a comprehensive configuration dictionary:

```python
config = {
    "model_config": {
        "base_model": "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "finetuned_model": "pradeep9322/llama3-Medical-Chat",
        "max_seq_length": 2048,
        "dtype": torch.float16,
        "load_in_4bit": True,
    },
    "lora_config": {
        "r": 16,                    # LoRA rank
        "lora_alpha": 16,          # LoRA scaling
        "lora_dropout": 0,         # Dropout rate
        "use_gradient_checkpointing": True,
    },
    "training_config": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
        "learning_rate": 2e-4,
        "warmup_steps": 5,
    }
}
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU with CUDA support (recommended for training)
- ~15GB GPU memory (with 4-bit quantization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HealthAI-Finetune.git
cd HealthAI-Finetune
```

2. Install required packages:
```bash
pip install torch transformers datasets huggingface-hub
pip install unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git
pip install trl peft wandb
pip install xformers
```

3. Set up Hugging Face authentication:
```bash
huggingface-cli login
```

### Running the Notebooks

#### Step 1: Prepare Dataset
Open and run `dataset-making.ipynb`:
- Loads medical data from Hugging Face Datasets
- Formats into Llama 3 chat template
- Uploads formatted dataset to your Hugging Face account

#### Step 2: Fine-tune Model
Open and run `finetune_llama3.ipynb`:
- Loads base Llama 3 model with 4-bit quantization
- Applies LoRA adapters for efficient training
- Trains on your formatted medical dataset
- Tests inference on sample medical queries
- Saves and uploads fine-tuned model to Hugging Face Hub

## 📈 Training Metrics

The training process tracks:
- **Memory Usage**: Before/after training GPU memory consumption
- **Training Stats**: Loss, learning rate changes, training time
- **Model Performance**: Inference quality on sample medical prompts

## 🏥 Dataset Information

**ChatDoctor-HealthCareMagic-100k Dataset**:
- Source: Hugging Face Datasets
- Size: 100k examples
- Format: Instruction, input (question), output (answer)
- Domain: Healthcare and medical Q&A
- Link to download: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k


## 🔍 Model Details

**Base Model**: Llama 3 8B Instruct (4-bit quantized)
- **Architecture**: 8B parameters
- **Context Length**: 2048 tokens
- **Quantization**: 4-bit (NF4)
- **Efficiency**: ~5GB VRAM with LoRA

**LoRA Configuration**:
- **Rank (r)**: 16 - Balance between quality and efficiency
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Alpha**: 16 - Scaling factor for LoRA updates

## 💾 Output Files

After fine-tuning, the following files are generated:
- `outputs/` - Training checkpoints and logs
- `trainer_stats.json` - Detailed training statistics
- Fine-tuned model uploaded to Hugging Face Hub

## 🔄 Usage in Production

To use the fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "pradeep9322/llama3-Medical-Chat"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare input
prompt = """<|start_header_id|>system<|end_header_id|>
You are a medical professional. Answer medical questions accurately.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is hypertension?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## ⚙️ Performance Optimization

The project uses several techniques for efficient training:

1. **4-bit Quantization**: Reduces model size by 75% with minimal quality loss
2. **LoRA Adapters**: Fine-tunes only 0.3% of parameters instead of all 8B
3. **Gradient Checkpointing**: Reduces memory usage by ~30%
4. **Packing**: Groups sequences efficiently (can be enabled)
5. **Unsloth Optimization**: 4x faster training without quality loss
