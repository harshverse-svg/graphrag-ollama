from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graphrag_ollama.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GRAPH_HTML_PATH,
    DEFAULT_GRAPH_JSON_PATH,
    DEFAULT_TEMPLATE_PATH,
)
from graphrag_ollama.graph_pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query the GraphRAG pipeline with Ollama.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH, help="Input CSV dataset.")
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH_JSON_PATH, help="Path for exported graph JSON.")
    parser.add_argument("--graph-html", type=Path, default=DEFAULT_GRAPH_HTML_PATH, help="Path for rendered graph HTML.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH, help="Visualization template HTML.")
    parser.add_argument("--question", type=str, default=None, help="Optional question to ask after the graph is built.")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional limit for quick tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    input_csv = args.input
    if input_csv == DEFAULT_DATASET_PATH and not input_csv.exists():
        from graphrag_ollama.app_support import get_active_dataset_path
        input_csv = get_active_dataset_path()
        print(f"Default dataset not found, falling back to: {input_csv}")

    run_pipeline(
        input_csv=input_csv,
        graph_json_path=args.graph_json,
        graph_html_path=args.graph_html,
        template_path=args.template,
        question=args.question,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
