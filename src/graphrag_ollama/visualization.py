from __future__ import annotations

import json
from pathlib import Path

from llama_index.core.graph_stores.types import EntityNode


def export_graph_data(graph_store, output_file: Path) -> dict:
    nx_graph = graph_store._to_networkx()

    node_meta: dict[str, dict] = {}
    for node in graph_store.graph.nodes.values():
        if isinstance(node, EntityNode):
            node_meta[node.id] = {
                "id": node.id,
                "label": node.name,
                "type": node.label if node.label else "OTHER",
                "description": node.properties.get("entity_description", ""),
            }

    nodes_data = [node_meta[node_id] for node_id in nx_graph.nodes() if node_id in node_meta]
    links_data: list[dict] = []
    for source, target, data in nx_graph.edges(data=True):
        links_data.append(
            {
                "source": source,
                "target": target,
                "label": data.get("relationship", ""),
                "description": data.get("description", ""),
            }
        )

    graph_data = {
        "nodes": nodes_data,
        "links": links_data,
        "communities": len(graph_store.get_community_summaries()),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    return graph_data


def visualize_graph(
    graph_data: dict,
    template_file: Path,
    output_file: Path,
) -> None:
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    html = template_file.read_text(encoding="utf-8")
    html = html.replace("GRAPH_DATA_PLACEHOLDER", json.dumps(graph_data))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
