"""Qwen2-VL vision-language model implementation."""
import json
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor

from .base import BaseLLM


class Qwen2VL(BaseLLM):
    """Qwen2-VL vision-language model implementation (~2B params).

    Qwen2-VL is a lightweight multimodal model from Alibaba that combines
    vision and language understanding. This implementation handles loading
    the model with proper configuration.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cpu"
    ):
        """
        Initialize Qwen2-VL model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        super().__init__(model_name, device)
        self.processor = None

    def load_model(self) -> None:
        """Load Qwen2-VL model with appropriate configuration."""
        try:
            # Load processor first
            print(f"Loading Qwen2-VL processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load the model
            print(f"Loading Qwen2-VL model...")
            try:
                # Try using the model's built-in loader
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
            except (ImportError, AttributeError):
                # Fallback: try with AutoModel
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )

            self.model.eval()
            print(f"Qwen2-VL model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading Qwen2-VL model: {e}")
            raise

    def process_image(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Process image with Qwen2-VL model.

        Args:
            image: PIL Image object
            prompt: User prompt for the model
            system_prompt: System prompt with instructions

        Returns:
            Dictionary with extracted data from the image
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Construct full prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            # Prepare input in Qwen2-VL's native format with messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ],
                }
            ]

            # Apply chat template - Qwen2-VL's native way of handling conversations
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except (AttributeError, TypeError):
                # Fallback if apply_chat_template not available
                text = full_prompt

            # Process: Try to use the processor with the messages directly
            try:
                # Qwen2-VL's preferred approach - pass messages directly
                inputs = self.processor(
                    messages,
                    padding=True,
                    return_tensors="pt"
                )
            except (TypeError, AttributeError):
                # Fallback: use standard text + image approach
                inputs = self.processor(
                    text=full_prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0
                )

            # Decode response
            # For messages-based input, the processor expects the full output_ids
            # For text+image, we need to skip input_ids
            try:
                # Try to extract only generated tokens
                input_token_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, input_token_len:]
                response = self.processor.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
            except (IndexError, AttributeError):
                # Fallback: decode full output
                response = self.processor.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )

            # Try to extract JSON from response
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    extracted_data = json.loads(json_str)
                else:
                    extracted_data = {"raw_response": response}
            except json.JSONDecodeError:
                extracted_data = {"raw_response": response}

            return extracted_data

        except Exception as e:
            print(f"Error processing image with Qwen2-VL: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.processor = None

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print("Qwen2-VL model unloaded")
