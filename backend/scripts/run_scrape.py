from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graphrag_ollama.config import DEFAULT_DATASET_PATH
from graphrag_ollama.scrape import run_scrape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape AI copyright sources into a CSV dataset.")
    parser.add_argument("--query", action="append", required=True, help="Search query to send to SerpAPI. Use multiple times.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="CSV file to write.")
    parser.add_argument("--num-results", type=int, default=10, help="Max organic search results per query.")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent article scrapers.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds before each article scrape.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scrape(
        queries=args.query,
        output_path=args.output,
        num_results=args.num_results,
        max_workers=args.max_workers,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
