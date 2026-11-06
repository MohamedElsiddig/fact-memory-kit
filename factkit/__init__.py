"""factkit — local-first fact memory with hybrid retrieval and provenance."""
__version__ = "0.1.0"

from .config import Config          # noqa: F401
from .embed import EmbeddingClient  # noqa: F401
from .hybrid import HybridSearch    # noqa: F401
from .store import Store            # noqa: F401

__all__ = ["Config", "EmbeddingClient", "HybridSearch", "Store", "__version__"]
