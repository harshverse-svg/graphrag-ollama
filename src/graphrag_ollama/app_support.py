from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GRAPH_HTML_PATH,
    DEFAULT_GRAPH_JSON_PATH,
    PROJECT_ROOT,
    load_runtime_config,
)
from .graph_pipeline import get_query_llm


SAMPLE_GRAPH_JSON_PATH = PROJECT_ROOT / "data" / "sample" / "graph_data.json"
SAMPLE_GRAPH_HTML_PATH = PROJECT_ROOT / "data" / "sample" / "ai_copyright_graph.html"
SAMPLE_DATASET_PATH = PROJECT_ROOT / "data" / "sample" / "ai_copyright_dataset.csv"


def ensure_env_file() -> Path:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    if not env_path.exists() and example_path.exists():
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
    return env_path


def get_active_graph_paths() -> tuple[Path, Path]:
    graph_json = DEFAULT_GRAPH_JSON_PATH if DEFAULT_GRAPH_JSON_PATH.exists() else SAMPLE_GRAPH_JSON_PATH
    graph_html = DEFAULT_GRAPH_HTML_PATH if DEFAULT_GRAPH_HTML_PATH.exists() else SAMPLE_GRAPH_HTML_PATH
    return graph_json, graph_html


def get_active_dataset_path() -> Path:
    return DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else SAMPLE_DATASET_PATH


def load_graph_data(path: Path | None = None) -> dict:
    target = path or get_active_graph_paths()[0]
    return json.loads(target.read_text(encoding="utf-8"))


def graph_stats(graph_data: dict) -> dict[str, int]:
    return {
        "nodes": len(graph_data.get("nodes", [])),
        "links": len(graph_data.get("links", [])),
        "communities": int(graph_data.get("communities", 0)),
    }


def build_graph_context(graph_data: dict, max_nodes: int = 40, max_edges: int = 60) -> str:
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])

    type_counts = Counter(node.get("type", "OTHER") for node in nodes)
    node_lookup = {node["id"]: node for node in nodes}
    degree_counter = Counter()
    for link in links:
        degree_counter[link["source"]] += 1
        degree_counter[link["target"]] += 1

    top_nodes = sorted(
        nodes,
        key=lambda item: degree_counter.get(item["id"], 0),
        reverse=True,
    )[:max_nodes]

    top_edges = links[:max_edges]

    type_summary = "\n".join(f"- {node_type}: {count}" for node_type, count in type_counts.most_common())
    node_summary = "\n".join(
        f"- {node.get('label', 'Unknown')} [{node.get('type', 'OTHER')}] "
        f"(degree {degree_counter.get(node['id'], 0)}): {node.get('description', '')}"
        for node in top_nodes
    )
    edge_summary = "\n".join(
        f"- {node_lookup.get(link['source'], {}).get('label', link['source'])} "
        f"--[{link.get('label', 'RELATED')}]--> "
        f"{node_lookup.get(link['target'], {}).get('label', link['target'])}: "
        f"{link.get('description', '')}"
        for link in top_edges
    )

    return (
        "Knowledge graph summary\n"
        f"Total nodes: {len(nodes)}\n"
        f"Total links: {len(links)}\n"
        f"Communities: {graph_data.get('communities', 0)}\n\n"
        "Entity type counts\n"
        f"{type_summary}\n\n"
        "Important entities\n"
        f"{node_summary}\n\n"
        "Important relationships\n"
        f"{edge_summary}"
    )


def answer_question_from_graph(question: str, graph_data: dict) -> str:
    runtime = load_runtime_config()
    llm = get_query_llm()
    context = build_graph_context(graph_data)
    prompt = (
        "You are answering questions about a GraphRAG knowledge graph focused on AI copyright and governance.\n"
        f"Ollama model: {runtime.query_model}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly in plain English. If the graph context is incomplete, say what is uncertain."
    )
    return llm.complete(prompt).text.strip()
