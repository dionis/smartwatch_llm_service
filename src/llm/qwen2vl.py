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

            # Prepare input in Qwen2-VL's native format
            # Qwen2-VL expects messages in a specific chat format
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

            # Apply chat template and process
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision info (this is key for Qwen2-VL)
            image_inputs, video_inputs = self.processor.process_vision_info(messages)

            # Prepare inputs with proper format
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
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

            # Decode response - extract only the generated part
            generated_ids = [
                output_ids[len(inputs["input_ids"][i]):]
                for i in range(len(inputs["input_ids"]))
            ]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

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
