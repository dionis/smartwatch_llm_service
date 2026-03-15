"""Factory for creating LLM model instances."""
from typing import Optional
import os

from .base import BaseLLM
from .phi4 import Phi4LLM
from .generic_multimodal import GenericMultimodalLLM
from .moondream import MoondreamLLM
from .ollama import OllamaLLM


class LLMFactory:
    """Factory class for creating LLM instances with flexibility."""

    # Predefined model configurations
    MODEL_CONFIGS = {
        "phi4": {
            "class": Phi4LLM,
            "model_name": "microsoft/phi-4",
        },
        "gemma": {
            "class": GenericMultimodalLLM,
            "model_name": "google/gemma-3-2b-vision",
            "use_vision2seq": False
        },
        "gemma3n": {
            "class": GenericMultimodalLLM,
            "model_name": "google/gemma-3n-vision",
            "use_vision2seq": False
        },
        "llava": {
            "class": GenericMultimodalLLM,
            "model_name": "llava-hf/llava-1.5-7b-hf",
            "use_vision2seq": True
        },
        "moondream": {
            "class": MoondreamLLM,
            "model_name": "vikhyatk/moondream2",
        },
        "ollama": {
            "class": OllamaLLM,
            "model_name": "phi4",
            "host": "http://localhost:11434"
        },
        "qwen2vl": {
            "class": GenericMultimodalLLM,
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "use_vision2seq": False
        }
    }

    @classmethod
    def create_llm(
        cls,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cpu"
    ) -> BaseLLM:
        """
        Create an LLM instance.

        Args:
            model_type: Predefined model type (phi4, gemma, gemma3n, llava, moondream, qwen2vl, ollama)
            model_name: Custom model name from HuggingFace
            device: Device to run inference on

        Returns:
            BaseLLM instance

        Example:
            # Use predefined model
            llm = LLMFactory.create_llm(model_type="phi4")

            # Use Ollama
            llm = LLMFactory.create_llm(model_type="ollama")

            # Use custom model
            llm = LLMFactory.create_llm(model_name="custom/model-name")
        """
        # Check environment variable for model override
        env_model_type = os.getenv("LLM_MODEL_TYPE")
        env_model_name = os.getenv("LLM_MODEL_NAME")

        if env_model_type:
            model_type = env_model_type
        if env_model_name:
            model_name = env_model_name

        # Use custom model name if provided
        if model_name:
            print(f"Creating custom model: {model_name}")
            return GenericMultimodalLLM(model_name=model_name, device=device)

        # Use predefined model type
        if model_type and model_type in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[model_type]
            model_class = config["class"]
            kwargs = {
                "model_name": config["model_name"],
                "device": device
            }
            if "use_vision2seq" in config:
                kwargs["use_vision2seq"] = config["use_vision2seq"]
            if "host" in config:
                # Check for environment variable override
                ollama_host = os.getenv("OLLAMA_HOST", config["host"])
                kwargs["host"] = ollama_host

            print(f"Creating predefined model: {model_type}")
            return model_class(**kwargs)

        # Default to phi4
        print("No model specified, defaulting to phi4")
        return Phi4LLM(device=device)

    @classmethod
    def list_available_models(cls) -> list:
        """List all predefined model types."""
        return list(cls.MODEL_CONFIGS.keys())
