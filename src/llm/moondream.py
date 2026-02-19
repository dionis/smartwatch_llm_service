"""Moondream2 vision-language model implementation."""
import json
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

from .base import BaseLLM


class MoondreamLLM(BaseLLM):
    """Moondream2 lightweight vision-language model (~1.8B params, ~4GB RAM)."""

    # Pin to a specific revision for stability
    DEFAULT_REVISION = "2025-06-21"

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

    def load_model(self) -> None:
        """Load the Moondream2 model."""
        try:
            # Moondream2 uses its own custom architecture via trust_remote_code
            # It does NOT use a separate processor — the model handles everything
            dtype = torch.float16 if "cuda" in self.device else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
                dtype=dtype,
                device_map={"": self.device}
            )
            # Moondream doesn't use a separate processor, set a dummy to satisfy BaseLLM
            self.processor = True
            print(f"Moondream2 model loaded successfully on {self.device} (revision: {self.revision})")
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
        Process image with Moondream2 model using its native query API.

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

            # Use Moondream's native query API
            result = self.model.query(image, full_prompt)
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)

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

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print("Moondream2 model unloaded")
