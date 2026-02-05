"""Gradio interface for testing FastAPI and gRPC services."""
import gradio as gr
import requests
import io
import json
from datetime import datetime
from PIL import Image
import grpc

# Import gRPC generated code
try:
    from ..grpc_service.generated import smartwatch_pb2, smartwatch_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    print("Warning: gRPC protobuf files not found. gRPC testing will be disabled.")
    print("Run: python main.py generate-protos")
    GRPC_AVAILABLE = False


def test_fastapi(image, timestamp, user_id, command, description, fastapi_url):
    """
    Test FastAPI service.

    Args:
        image: PIL Image object
        timestamp: Request timestamp
        user_id: User ID
        command: Command
        description: Description
        fastapi_url: FastAPI server URL

    Returns:
        Formatted response string
    """
    try:
        if image is None:
            return "Error: Please upload an image"

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Prepare form data
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'command': command,
            'description': description
        }

        # Make request to FastAPI
        response = requests.post(
            f"{fastapi_url}/process",
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return format_response(result, "FastAPI")
        else:
            return f"Error: HTTP {response.status_code}\n{response.text}"

    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to FastAPI server at {fastapi_url}\nMake sure the server is running: uv run python main.py fastapi"
    except Exception as e:
        return f"Error calling FastAPI: {str(e)}"


def test_grpc(image, timestamp, user_id, command, description, grpc_host, grpc_port):
    """
    Test gRPC service.

    Args:
        image: PIL Image object
        timestamp: Request timestamp
        user_id: User ID
        command: Command
        description: Description
        grpc_host: gRPC server host
        grpc_port: gRPC server port

    Returns:
        Formatted response string
    """
    if not GRPC_AVAILABLE:
        return "Error: gRPC protobuf files not generated.\nRun: uv run python main.py generate-protos"

    try:
        if image is None:
            return "Error: Please upload an image"

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()

        # Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}")
        stub = smartwatch_pb2_grpc.SmartwatchServiceStub(channel)

        # Create request
        request = smartwatch_pb2.SmartwatchRequest(
            image=image_bytes,
            timestamp=timestamp,
            user_id=user_id,
            command=command,
            description=description
        )

        # Make gRPC call
        response = stub.ProcessSmartwatchImage(request, timeout=60)

        # Convert response to dict
        result = {
            'success': response.success,
            'extracted_data': json.loads(response.extracted_data) if response.extracted_data else {},
            'inference_time_ms': response.inference_time_ms,
            'error_message': response.error_message if response.error_message else None
        }

        channel.close()
        return format_response(result, "gRPC")

    except grpc.RpcError as e:
        return f"Error: Could not connect to gRPC server at {grpc_host}:{grpc_port}\nMake sure the server is running: uv run python main.py grpc\nDetails: {e.details()}"
    except Exception as e:
        return f"Error calling gRPC: {str(e)}"


def format_response(result, service_type):
    """
    Format the response for display.

    Args:
        result: Response dictionary
        service_type: Type of service (FastAPI or gRPC)

    Returns:
        Formatted string
    """
    output = f"=== {service_type} Response ===\n\n"
    output += f"Success: {result.get('success', False)}\n"
    output += f"Inference Time: {result.get('inference_time_ms', 0)} ms\n\n"

    if result.get('error_message'):
        output += f"Error Message: {result['error_message']}\n\n"

    output += "Extracted Data:\n"
    extracted_data = result.get('extracted_data', {})
    if isinstance(extracted_data, dict):
        output += json.dumps(extracted_data, indent=2, ensure_ascii=False)
    else:
        output += str(extracted_data)

    return output


def create_interface():
    """Create and configure the Gradio interface."""

    # Default values
    default_timestamp = datetime.now().isoformat()
    default_user_id = "user123"
    default_command = "extract_metrics"
    default_description = "Extract all visible health metrics from this smartwatch screenshot"

    with gr.Blocks(title="Smartwatch LLM Service - Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔬 Smartwatch LLM Service Tester

            Upload a smartwatch screenshot and test both FastAPI and gRPC services.
            Make sure the respective servers are running before testing:
            - FastAPI: `uv run python main.py fastapi`
            - gRPC: `uv run python main.py grpc`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                # Image input
                image_input = gr.Image(
                    label="Smartwatch Screenshot",
                    type="pil",
                    height=300
                )

                # Metadata inputs
                timestamp_input = gr.Textbox(
                    label="Timestamp",
                    value=default_timestamp,
                    placeholder="ISO format timestamp"
                )

                user_id_input = gr.Textbox(
                    label="User ID",
                    value=default_user_id,
                    placeholder="User identifier"
                )

                command_input = gr.Textbox(
                    label="Command",
                    value=default_command,
                    placeholder="Command to execute"
                )

                description_input = gr.Textbox(
                    label="Description",
                    value=default_description,
                    placeholder="Task description",
                    lines=3
                )

            with gr.Column(scale=1):
                gr.Markdown("### Service Configuration")

                # Service selection
                service_type = gr.Radio(
                    choices=["FastAPI", "gRPC"],
                    label="Service Type",
                    value="FastAPI"
                )

                # FastAPI configuration
                with gr.Group(visible=True) as fastapi_config:
                    gr.Markdown("#### FastAPI Configuration")
                    fastapi_url = gr.Textbox(
                        label="FastAPI URL",
                        value="http://localhost:8000",
                        placeholder="http://localhost:8000"
                    )

                # gRPC configuration
                with gr.Group(visible=False) as grpc_config:
                    gr.Markdown("#### gRPC Configuration")
                    grpc_host = gr.Textbox(
                        label="gRPC Host",
                        value="localhost",
                        placeholder="localhost"
                    )
                    grpc_port = gr.Number(
                        label="gRPC Port",
                        value=50051,
                        precision=0
                    )

                # Test button
                test_button = gr.Button("Test Service", variant="primary", size="lg")

                # Output
                gr.Markdown("### Response")
                output_text = gr.Textbox(
                    label="Service Response",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )

        # Toggle visibility based on service type
        def update_config_visibility(service):
            if service == "FastAPI":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        service_type.change(
            fn=update_config_visibility,
            inputs=[service_type],
            outputs=[fastapi_config, grpc_config]
        )

        # Handle test button click
        def handle_test(service, image, timestamp, user_id, command, description,
                       fastapi_url, grpc_host, grpc_port):
            if service == "FastAPI":
                return test_fastapi(image, timestamp, user_id, command, description, fastapi_url)
            else:
                return test_grpc(image, timestamp, user_id, command, description,
                               grpc_host, int(grpc_port))

        test_button.click(
            fn=handle_test,
            inputs=[
                service_type, image_input, timestamp_input, user_id_input,
                command_input, description_input, fastapi_url, grpc_host, grpc_port
            ],
            outputs=[output_text]
        )

        # Examples
        gr.Markdown("### Tips")
        gr.Markdown(
            """
            - Upload a clear screenshot of a smartwatch display
            - The LLM will extract metrics like heart rate, steps, calories, etc.
            - Compare response times between FastAPI and gRPC
            - Make sure servers are running before testing
            """
        )

    return demo


def launch(server_name="0.0.0.0", server_port=7860, share=False):
    """
    Launch the Gradio interface.

    Args:
        server_name: Server host
        server_port: Server port
        share: Whether to create a public link
    """
    demo = create_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    launch()
