from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ai_copyright_dataset.csv"
DEFAULT_GRAPH_JSON_PATH = PROJECT_ROOT / "data" / "graph_data.json"
DEFAULT_GRAPH_HTML_PATH = PROJECT_ROOT / "data" / "ai_copyright_graph.html"
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "visualization" / "graph_template.html"


@dataclass(slots=True)
class RuntimeConfig:
    serpapi_key: str | None
    ollama_base_url: str
    extraction_model: str
    query_model: str
    request_timeout: float
    context_window: int


def load_runtime_config() -> RuntimeConfig:
    load_dotenv()
    return RuntimeConfig(
        serpapi_key=os.getenv("SERPAPI_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        extraction_model=os.getenv("OLLAMA_EXTRACTION_MODEL", "gemma4:latest"),
        query_model=os.getenv("OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_EXTRACTION_MODEL", "gemma4:latest")),
        request_timeout=float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")),
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
    )
