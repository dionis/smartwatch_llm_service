"""Moondream2 vision-language model implementation."""
import json
import os
import sys
from typing import Dict, Any
from PIL import Image
import torch

from .base import BaseLLM


class MoondreamLLM(BaseLLM):
    """Moondream2 lightweight vision-language model (~1.8B params, ~4GB RAM).

    This implementation loads the model directly using Moondream's native
    MoondreamModel class instead of AutoModelForCausalLM, which avoids
    compatibility issues with newer versions of transformers (5.x+) that
    require 'all_tied_weights_keys' on PreTrainedModel subclasses.
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
        """Load Moondream2 using its native MoondreamModel class directly.

        This bypasses AutoModelForCausalLM / PreTrainedModel to avoid the
        'all_tied_weights_keys' incompatibility with transformers >= 5.x.
        """
        try:
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file as load_safetensors

            dtype = torch.float16 if "cuda" in self.device else torch.float32

            # Step 1: Download the model repo to a local cache directory
            print(f"Downloading Moondream2 (revision: {self.revision})...")
            model_path = snapshot_download(
                self.model_name,
                revision=self.revision,
            )
            print(f"Model files cached at: {model_path}")

            # Step 2: Add the model directory to sys.path so we can import
            #         moondream's own modules (config, moondream, vision, etc.)
            if model_path not in sys.path:
                sys.path.insert(0, model_path)

            # Step 3: Import moondream's native classes directly
            from config import MoondreamConfig
            from moondream import MoondreamModel

            # Step 4: Create the model with default config and load weights
            config = MoondreamConfig()
            self.model = MoondreamModel(config, dtype=dtype)

            # Load safetensors weights
            weights_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(weights_file):
                state_dict = load_safetensors(weights_file)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"model.safetensors not found in {model_path}"
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            # The MoondreamModel has its own tokenizer
            self.tokenizer = self.model.tokenizer

            # Set processor for BaseLLM compatibility
            self.processor = self.tokenizer

            print(f"Moondream2 model loaded successfully on {self.device} "
                  f"(revision: {self.revision})")

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
