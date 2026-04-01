"""
HuggingFace Spaces entry point.
Redirects to the FastAPI server defined in server.py.
"""
from server import app  # noqa: F401 — re-export for uvicorn/gunicorn
