"""Main entry point for the smartwatch LLM service."""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_fastapi():
    """Run FastAPI server."""
    import uvicorn
    from src.api.fastapi_server import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


def run_grpc():
    """Run gRPC server."""
    from src.grpc_service.server import serve
    serve(port=50051)


def generate_protos():
    """Generate protobuf files."""
    from src.grpc_service.generate_protos import generate_grpc_code
    success = generate_grpc_code()
    sys.exit(0 if success else 1)


def run_gradio():
    """Run Gradio interface."""
    from src.ui.gradio_interface import launch
    launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smartwatch LLM Service",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # FastAPI command
    subparsers.add_parser("fastapi", help="Run FastAPI server")

    # gRPC command
    subparsers.add_parser("grpc", help="Run gRPC server")

    # Generate protos command
    subparsers.add_parser("generate-protos", help="Generate protobuf files")

    # Gradio command
    subparsers.add_parser("gradio", help="Run Gradio UI interface")

    args = parser.parse_args()

    if args.command == "fastapi":
        print("Starting FastAPI server...")
        run_fastapi()
    elif args.command == "grpc":
        print("Starting gRPC server...")
        run_grpc()
    elif args.command == "generate-protos":
        print("Generating protobuf files...")
        generate_protos()
    elif args.command == "gradio":
        print("Starting Gradio interface...")
        run_gradio()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
