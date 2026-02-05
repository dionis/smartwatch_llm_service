# Smartwatch LLM Service

Backend service for processing smartwatch screenshots using multimodal LLMs. The service provides **FastAPI** (REST), **gRPC**, and **Gradio** (Web UI) interfaces for extracting data from smartwatch images.

## Features

- 🚀 **Triple Interface**: FastAPI (REST), gRPC, and Gradio (Web UI) support
- 🎨 **User-Friendly UI**: Gradio web interface for easy testing of both services
- 🧠 **Flexible LLM Backend**: Support for multiple multimodal models (Phi-4, Gemma, Gemma 3n, LLaVA, etc.)
- 🔄 **Easy Model Switching**: Change models via environment variables without code changes
- ⚡ **Lightning.ai Ready**: Configured for deployment on Lightning.ai platform
- 🧪 **Test Clients**: Comprehensive test clients for both FastAPI and gRPC
- 📊 **Inference Tracking**: Built-in timing for LLM inference

## Supported Models

### HuggingFace Models
- **Phi-4** (microsoft/phi-4) - ~7B params
- **Gemma** (google/gemma-3-2b-vision) - ~2B params
- **Gemma 3n** (google/gemma-3n-vision)
- **LLaVA** (llava-hf/llava-1.5-7b-hf) - ~7B params
- **Moondream** (vikhyatk/moondream2) - ~1.8B params
- Custom models via HuggingFace model identifiers

### Ollama Models (Recommended)
- **Phi-4** - Optimized version via Ollama
- **LLaVA** - Various sizes (7B, 13B)
- **BakLLaVA** - LLaVA with Mistral backbone
- Any Ollama-compatible vision model

## Quick Start

### 1. Install Dependencies

```bash
# Quick setup script
bash quick_start.sh

# Or manually with uv
uv sync
uv run python main.py generate-protos
```

### 2. Install Models

**Option A: Automated Installation (Recommended)**

Linux/macOS:
```bash
bash scripts/install_models.sh
```

Windows (PowerShell):
```powershell
.\scripts\install_models.ps1
```

**Option B: Manual Installation**

For Ollama (recommended for faster inference):
```bash
# Install Ollama: https://ollama.ai/download
bash scripts/setup_ollama.sh

# Configure
echo "LLM_MODEL_TYPE=ollama" >> .env
echo "LLM_MODEL_NAME=phi4" >> .env
```

For HuggingFace models:
```bash
# Download specific model
uv run python scripts/download_huggingface_model.py --model phi4

# Configure
echo "LLM_MODEL_TYPE=phi4" >> .env
echo "DEVICE=cpu" >> .env  # or cuda
```

See [scripts/README.md](scripts/README.md) for detailed model installation guide.

### 3. Run the Service

```bash
# Start FastAPI server
uv run python main.py fastapi

# Or start Gradio UI for testing
uv run python main.py gradio
```

## Installation

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Generate gRPC protobuf files
uv run python main.py generate-protos
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Generate gRPC protobuf files
python main.py generate-protos
```

## Usage

### Running FastAPI Server

```bash
# Using uv
uv run python main.py fastapi

# Or directly with uvicorn
uv run uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000

# Access interactive docs at: http://localhost:8000/docs
```

### Running gRPC Server

```bash
# Using uv
uv run python main.py grpc

# Server will start on port 50051
```

### Running Gradio UI

```bash
# Using uv
uv run python main.py gradio

# Access the web interface at: http://localhost:7860
```

The Gradio interface provides an easy-to-use web UI for testing both FastAPI and gRPC services. Simply upload a smartwatch screenshot and choose which service to test.

### Running on Lightning.ai

```bash
# Deploy to Lightning.ai
lightning run app lightning_app.py --cloud
```

## Configuration

Create a `.env` file (see `.env.example` for template):

### Environment Variables

```bash
# Model configuration
LLM_MODEL_TYPE=ollama        # ollama, phi4, gemma, gemma3n, llava, moondream
LLM_MODEL_NAME=phi4          # Model name (for ollama: phi4, llava, bakllava, etc.)

# Ollama configuration (only for LLM_MODEL_TYPE=ollama)
OLLAMA_HOST=http://localhost:11434

# Device configuration (not used for ollama)
DEVICE=cpu                   # Use 'cuda' for GPU

# Service ports
FASTAPI_PORT=8000
GRPC_PORT=50051
```

### Changing Models

You can change models in three ways:

1. **Environment Variables** (Recommended)
```bash
# For Ollama
export LLM_MODEL_TYPE=ollama
export LLM_MODEL_NAME=phi4

# For HuggingFace models
export LLM_MODEL_TYPE=gemma3n
export DEVICE=cpu
```

2. **Code Configuration**
```python
from src.llm.factory import LLMFactory
llm = LLMFactory.create_llm(model_type="phi4", device="cpu")
```

3. **Custom Model**
```bash
export LLM_MODEL_NAME="custom/model-identifier"
```

### Ollama vs HuggingFace

| Feature | Ollama | HuggingFace |
|---------|--------|-------------|
| Speed | ⚡ Faster (optimized) | Slower |
| Memory | 💾 Lower (quantized) | Higher |
| Setup | ✅ Simple | Auto-download |
| Models | Limited to Ollama catalog | Full HuggingFace access |

**Recommendation**: Use Ollama for production, HuggingFace for experimentation.

## Testing

### Option 1: Gradio Web UI (Easiest)

```bash
# Start the Gradio interface
uv run python main.py gradio

# Open http://localhost:7860 in your browser
# Upload an image and test both FastAPI and gRPC services
```

### Option 2: Command-Line Clients

#### Test FastAPI Service

```bash
# Health check and model info
uv run python client/test_fastapi_client.py

# Test with image
uv run python client/test_fastapi_client.py path/to/smartwatch_image.png
```

#### Test gRPC Service

```bash
# Test with image
uv run python client/test_grpc_client.py path/to/smartwatch_image.png
```

## API Examples

### FastAPI (REST)

```bash
curl -X POST "http://localhost:8000/process" \
  -F "image=@smartwatch.png" \
  -F "timestamp=2026-02-01T10:30:00" \
  -F "user_id=user123" \
  -F "command=extract_metrics" \
  -F "description=Extract health metrics"
```

### gRPC

See `client/test_grpc_client.py` for example implementation.

## Project Structure

```
curso_platzy/
├── src/
│   ├── api/                 # FastAPI implementation
│   │   ├── fastapi_server.py
│   │   └── models.py
│   ├── grpc_service/        # gRPC implementation
│   │   ├── server.py
│   │   ├── protos/          # Protocol buffer definitions
│   │   └── generated/       # Generated gRPC code
│   ├── llm/                 # LLM implementations
│   │   ├── base.py          # Abstract base class
│   │   ├── phi4.py          # Phi-4 implementation
│   │   ├── generic_multimodal.py
│   │   └── factory.py       # Model factory
│   ├── ui/                  # User interface
│   │   └── gradio_interface.py  # Gradio web UI
│   └── utils/               # Utility functions
├── client/                  # Test clients
│   ├── test_fastapi_client.py
│   └── test_grpc_client.py
├── tests/                   # Unit tests
├── lightning_app.py         # Lightning.ai configuration
├── main.py                  # Main entry point
└── pyproject.toml          # Dependencies
```

## Development

### Generate Protobuf Files

After modifying `.proto` files:

```bash
uv run python main.py generate-protos
```

### Adding New Models

1. Create a new class in `src/llm/` that extends `BaseLLM`
2. Add configuration to `LLMFactory.MODEL_CONFIGS`
3. Test with both FastAPI and gRPC clients

## Response Format

Both services return JSON with the following structure:

```json
{
  "success": true,
  "extracted_data": {
    "heart_rate": "72 bpm",
    "steps": "8543",
    "calories": "450 kcal"
  },
  "inference_time_ms": 1250,
  "error_message": null
}
```

## Requirements

- Python >= 3.11
- PyTorch (CPU or CUDA)
- FastAPI + Uvicorn
- gRPC + gRPC Tools
- Gradio (for Web UI)
- Transformers
- Lightning (for deployment)
- Ollama (optional, for optimized inference)

## Model Installation Guide

For detailed information about installing and managing models, see the [Model Installation Guide](scripts/README.md).

### Quick Reference

**Install all tools and models:**
```bash
# Linux/macOS
bash scripts/install_models.sh

# Windows
.\scripts\install_models.ps1
```

**Download specific HuggingFace model:**
```bash
uv run python scripts/download_huggingface_model.py --model phi4
```

**Setup Ollama:**
```bash
bash scripts/setup_ollama.sh
```

**List available models:**
```bash
# HuggingFace
uv run python scripts/download_huggingface_model.py --list

# Ollama
ollama list
```

## Troubleshooting

### Model Download Issues
- **Slow downloads**: HuggingFace models are large (4-14GB). Consider using Ollama for faster setup.
- **Out of memory**: Try smaller models (moondream, gemma) or use Ollama's quantized versions.
- **CUDA errors**: Set `DEVICE=cpu` in `.env` if GPU is unavailable.

### Ollama Issues
- **Connection refused**: Ensure Ollama is running with `ollama serve`
- **Model not found**: Pull the model first with `ollama pull phi4`

### gRPC Errors
- **Import errors**: Regenerate protobuf files with `uv run python main.py generate-protos`

## Performance Tips

1. **Use Ollama for production**: 2-3x faster inference with lower memory usage
2. **Choose appropriate model size**:
   - Development/Testing: moondream (~1.8B)
   - Balanced: phi4, llava (~7B)
   - High accuracy: llava:13b (~13B)
3. **Use GPU when available**: Set `DEVICE=cuda` (HuggingFace only)
4. **Pre-download models**: Avoid delays on first request

## License

This project is part of a Platzi course on AI backend development.
