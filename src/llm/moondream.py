"""Moondream2 vision-language model implementation."""
import json
import os
import sys
from typing import Dict, Any
import importlib.util
from PIL import Image
import torch

from .base import BaseLLM


class MoondreamLLM(BaseLLM):
    """Moondream2 lightweight vision-language model (~1.8B params, ~4GB RAM).

    This implementation loads the model manually using a 'virtual package'
    strategy. This avoids both:
    1. 'attempted relative import' errors (by registering the snapshot
       as a proper Python package in sys.modules).
    2. 'all_tied_weights_keys' errors in newer transformers (by bypassing
       AutoModelForCausalLM entirely).
    """

    DEFAULT_REVISION = "2025-01-09"

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: str = "cpu",
        revision: str = None
    ):
        """
        Initialize Moondream2 model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu', 'cuda', 'mps')
            revision: Model revision to use (pinned for stability)
        """
        super().__init__(model_name, device)
        self.revision = revision or self.DEFAULT_REVISION
        self.tokenizer = None

    def load_model(self) -> None:
        """Load Moondream2 using a virtual package strategy.
        
        This is the most robust way to load Moondream2 as it bypasses
        problematic logic in transformers while ensuring remote code
        imports resolve correctly.
        """
        try:
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file as load_safetensors

            dtype = torch.float16 if "cuda" in self.device else torch.float32

            # Step 1: Download the model snapshot
            print(f"Loading Moondream2 (revision: {self.revision})...")
            model_path = snapshot_download(
                self.model_name,
                revision=self.revision,
            )
            print(f"Model files cached at: {model_path}")

            # Step 2: Create a Virtual Package
            # We register the model directory as a package named 'moondream_package'
            # to allow relative imports inside the model code to work properly.
            package_name = "moondream_repo"
            
            # Setup the module spec
            init_file = os.path.join(model_path, "__init__.py")
            # If __init__.py doesn't exist, we just need a dummy spec for the directory
            spec = importlib.util.spec_from_file_location(
                package_name, 
                init_file if os.path.exists(init_file) else os.path.join(model_path, "moondream.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            
            # Also add to sys.path to support absolute imports of submodules
            if model_path not in sys.path:
                sys.path.insert(0, model_path)

            # Step 3: Import classes from the snapshot directory
            # We use importlib to load specifically from the snapshot path
            def load_from_snapshot(name, filename):
                m_spec = importlib.util.spec_from_file_location(
                    f"{package_name}.{name}", 
                    os.path.join(model_path, filename)
                )
                m = importlib.util.module_from_spec(m_spec)
                # This is CRITICAL for relative imports:
                m.__package__ = package_name 
                sys.modules[f"{package_name}.{name}"] = m
                m_spec.loader.exec_module(m)
                return m

            config_mod = load_from_snapshot("config", "config.py")
            model_mod = load_from_snapshot("moondream", "moondream.py")

            # Load the vision module as well since moondream.py depends on it relatively
            load_from_snapshot("vision", "vision.py")

            # Step 4: Instantiate and load weights
            config = config_mod.MoondreamConfig()
            self.model = model_mod.MoondreamModel(config, dtype=dtype)

            weights_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(weights_file):
                state_dict = load_safetensors(weights_file)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"model.safetensors not found in {model_path}")

            self.model = self.model.to(self.device).eval()
            self.tokenizer = self.model.tokenizer
            self.processor = self.tokenizer

            print(f"Moondream2 model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading Moondream2 model: {e}")
            raise

    def process_image(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Process image with Moondream2 model.

        Args:
            image: PIL Image object
            prompt: User prompt for the model
            system_prompt: System prompt with instructions

        Returns:
            Dictionary with extracted data from the image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Construct full prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            # Use moondream's native query API
            result = self.model.query(image, full_prompt)
            answer = result.get("answer", "").strip()

            # Try to extract JSON from the answer
            try:
                start_idx = answer.find('{')
                end_idx = answer.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = answer[start_idx:end_idx]
                    extracted_data = json.loads(json_str)
                else:
                    extracted_data = {"raw_response": answer}
            except json.JSONDecodeError:
                extracted_data = {"raw_response": answer}

            return extracted_data

        except Exception as e:
            print(f"Error processing image with Moondream2: {e}")
            return {"error": str(e)}

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.processor = None
            self.tokenizer = None

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print("Moondream2 model unloaded")
