"""FastAPI server for smartwatch image processing."""
import time
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from .models import SmartwatchRequest, SmartwatchResponse, HealthCheckResponse
from ..llm.factory import LLMFactory


# Initialize FastAPI app
app = FastAPI(
    title="Smartwatch LLM Service",
    description="API for processing smartwatch screenshots using multimodal LLMs",
    version="0.1.0"
)

# Global LLM instance
llm_instance = None

# System prompt for the LLM
SYSTEM_PROMPT = """You are an AI assistant specialized in extracting data from smartwatch screenshots.
Your task is to analyze the image and extract all relevant information such as:
- Heart rate
- Steps count
- Calories burned
- Distance traveled
- Active minutes
- Sleep data
- Any other metrics visible

Return the data in a valid JSON format with clear key-value pairs.
Example: {"heart_rate": "72 bpm", "steps": "8543", "calories": "450 kcal"}
"""


@app.on_event("startup")
async def startup_event():
    """Initialize the LLM model on startup."""
    global llm_instance
    try:
        print("Loading LLM model...")
        llm_instance = LLMFactory.create_llm(
            model_type="phi4",  # Default model, can be overridden via env vars
            device="cpu"  # Change to "cuda" if GPU is available
        )
        llm_instance.load_model()
        print("LLM model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load LLM model on startup: {e}")
        print("Model will be loaded on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global llm_instance
    if llm_instance:
        llm_instance.unload_model()
        print("LLM model unloaded")


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=llm_instance is not None and llm_instance.model is not None,
        model_name=llm_instance.model_name if llm_instance else "Not loaded"
    )


@app.post("/process", response_model=SmartwatchResponse)
async def process_smartwatch_image(
    image: UploadFile = File(..., description="Smartwatch screenshot"),
    timestamp: str = Form(..., description="Hora de la petición"),
    user_id: str = Form(..., description="Usuario que realiza la petición"),
    command: str = Form(..., description="Orden o comando"),
    description: str = Form(..., description="Texto de descripción")
):
    """
    Process smartwatch image and extract data using LLM.

    Args:
        image: Uploaded image file
        timestamp: Request timestamp
        user_id: User making the request
        command: Command/order
        description: Description text

    Returns:
        SmartwatchResponse with extracted data
    """
    global llm_instance

    # Load model if not already loaded
    if llm_instance is None or llm_instance.model is None:
        try:
            print("Loading LLM model (lazy loading)...")
            llm_instance = LLMFactory.create_llm(model_type="phi4", device="cpu")
            llm_instance.load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load LLM model: {str(e)}"
            )

    try:
        # Read and validate image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Construct user prompt
        user_prompt = f"""Command: {command}
Description: {description}
User: {user_id}
Timestamp: {timestamp}

Analyze this smartwatch screenshot and extract all visible data.
Return the extracted data as a JSON object."""

        # Start inference timer
        start_time = time.time()

        # Process image with LLM
        extracted_data = llm_instance.process_image(
            image=pil_image,
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        # Calculate inference time
        inference_time_ms = int((time.time() - start_time) * 1000)

        # Check for errors in extraction
        if "error" in extracted_data:
            return SmartwatchResponse(
                success=False,
                extracted_data=extracted_data,
                inference_time_ms=inference_time_ms,
                error_message=extracted_data["error"]
            )

        return SmartwatchResponse(
            success=True,
            extracted_data=extracted_data,
            inference_time_ms=inference_time_ms,
            error_message=None
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List available predefined models."""
    return {
        "available_models": LLMFactory.list_available_models(),
        "current_model": llm_instance.model_name if llm_instance else None
    }
