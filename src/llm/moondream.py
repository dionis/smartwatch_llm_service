"""Moondream2 vision-language model implementation."""
import json
import os
import sys
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        """Load Moondream2 using transformers Auto classes.
        
        This handles remote code and package contexts correctly, avoiding 
        relative import issues.
        """
        try:
            print(f"Loading Moondream2 (revision: {self.revision})...")
            
            # Using AutoModelForCausalLM is the standard way and handles trust_remote_code correctly
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True
            )

            self.model.eval()

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
