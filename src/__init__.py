"""Top-level package.

Re-exports the primary Python API so users can do:

    from src import train, evaluate
"""
from .api import evaluate, train

__all__ = ["train", "evaluate"]
__version__ = "0.1.0"
