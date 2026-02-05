"""gRPC service module for smartwatch image processing."""
from .server import serve, SmartwatchServicer

__all__ = ["serve", "SmartwatchServicer"]
