from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import nest_asyncio
import networkx as nx
import pandas as pd
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.llms.ollama import Ollama
from pydantic import Field, field_validator

from .config import (
    DEFAULT_GRAPH_HTML_PATH,
    DEFAULT_GRAPH_JSON_PATH,
    DEFAULT_TEMPLATE_PATH,
    RuntimeConfig,
    load_runtime_config,
)
from .ontology import ExtractionResult, build_triplet_prompt
from .visualization import export_graph_data, visualize_graph


nest_asyncio.apply()

_EXTRACTION_LLM: LLM | None = None
_QUERY_LLM: LLM | None = None

MAX_PATHS_PER_CHUNK = 20
NUM_WORKERS = 4
MAX_CLUSTER_SIZE = 10


@dataclass(slots=True)
class ClusterAssignment:
    node: str
    cluster: int


def configure_models(config: RuntimeConfig | None = None) -> tuple[LLM, LLM]:
    runtime = config or load_runtime_config()
    extraction_llm = Ollama(
        model=runtime.extraction_model,
        base_url=runtime.ollama_base_url,
        request_timeout=runtime.request_timeout,
        context_window=runtime.context_window,
        is_function_calling_model=False,
    )
    query_llm = Ollama(
        model=runtime.query_model,
        base_url=runtime.ollama_base_url,
        request_timeout=runtime.request_timeout,
        context_window=runtime.context_window,
        is_function_calling_model=False,
    )

    global _EXTRACTION_LLM, _QUERY_LLM
    _EXTRACTION_LLM = extraction_llm
    _QUERY_LLM = query_llm
    Settings.llm = extraction_llm
    return extraction_llm, query_llm


def get_extraction_llm() -> LLM:
    if _EXTRACTION_LLM is None:
        configure_models()
    if _EXTRACTION_LLM is None:
        raise RuntimeError("Extraction model could not be initialized.")
    return _EXTRACTION_LLM


def get_query_llm() -> LLM:
    if _QUERY_LLM is None:
        configure_models()
    if _QUERY_LLM is None:
        raise RuntimeError("Query model could not be initialized.")
    return _QUERY_LLM


def _clean_json_payload(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start : end + 1]

    return cleaned.strip()


class GraphRAGExtractor(TransformComponent):
    llm: LLM = Field(default_factory=get_extraction_llm)
    extract_prompt: PromptTemplate = Field(default_factory=lambda: PromptTemplate(build_triplet_prompt()))
    num_workers: int = NUM_WORKERS
    max_paths_per_chunk: int = MAX_PATHS_PER_CHUNK

    @field_validator("extract_prompt", mode="before")
    @classmethod
    def coerce_to_prompt_template(cls, value):
        return PromptTemplate(value) if isinstance(value, str) else value

    def __call__(self, nodes, show_progress: bool = False, **kwargs):
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    def _fallback_extract(self, text: str) -> ExtractionResult:
        prompt_text = self.extract_prompt.format(
            text=text,
            max_knowledge_triplets=self.max_paths_per_chunk,
        )
        schema = json.dumps(ExtractionResult.model_json_schema(), indent=2)
        raw_response = self.llm.complete(
            f"{prompt_text}\n\nReturn only valid JSON matching this schema:\n{schema}\n"
        )
        return ExtractionResult.model_validate_json(
            _clean_json_payload(raw_response.text)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        text = node.get_content(metadata_mode="llm")
        try:
            if getattr(self.llm, "is_function_calling_model", False):
                result = await self.llm.astructured_predict(
                    ExtractionResult,
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )
            else:
                result = self._fallback_extract(text)
            entities = result.entities
            relationships = result.relationships
        except Exception as exc:
            print(f"Extraction error: {exc}")
            entities, relationships = [], []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        base_metadata = node.metadata.copy()

        existing_nodes += [
            EntityNode(
                name=entity.name,
                label=entity.type,
                properties={**base_metadata, "entity_description": entity.description},
            )
            for entity in entities
        ]

        entity_lookup = {entity.name: entity.type for entity in entities}
        for rel in relationships:
            source_node = EntityNode(
                name=rel.source,
                label=entity_lookup.get(rel.source, "ENTITY"),
                properties=base_metadata,
            )
            target_node = EntityNode(
                name=rel.target,
                label=entity_lookup.get(rel.target, "ENTITY"),
                properties=base_metadata,
            )
            if rel.source not in entity_lookup:
                existing_nodes.append(source_node)
            if rel.target not in entity_lookup:
                existing_nodes.append(target_node)

            existing_relations.append(
                Relation(
                    label=rel.relation,
                    source_id=source_node.id,
                    target_id=target_node.id,
                    properties={**base_metadata, "relationship_description": rel.description},
                )
            )

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(self, nodes, show_progress: bool = False, **kwargs):
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting triplets",
        )


class GraphRAGStore(SimplePropertyGraphStore):
    community_summaries: dict = {}

    def build_communities(self):
        print("Running community detection")
        nx_graph = self._to_networkx()
        if not nx_graph.nodes:
            print("Graph is empty, so no communities were detected.")
            return

        clusters = self._cluster_graph(nx_graph)
        community_info = self._collect_community_info(nx_graph, clusters)
        self._generate_summaries(community_info)

    def _cluster_graph(self, nx_graph: nx.Graph) -> list[ClusterAssignment]:
        communities = nx.community.louvain_communities(nx_graph, seed=42)
        assignments: list[ClusterAssignment] = []
        cluster_id = 0

        for community in communities:
            members = sorted(community)
            for start in range(0, len(members), MAX_CLUSTER_SIZE):
                chunk = members[start : start + MAX_CLUSTER_SIZE]
                assignments.extend(
                    ClusterAssignment(node=node_id, cluster=cluster_id)
                    for node_id in chunk
                )
                cluster_id += 1

        return assignments

    def _to_networkx(self) -> nx.Graph:
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            if isinstance(node, EntityNode):
                nx_graph.add_node(node.id)

        for relation in self.graph.relations.values():
            if relation.source_id in nx_graph and relation.target_id in nx_graph:
                nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relationship=relation.label,
                    description=relation.properties.get("relationship_description", ""),
                )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters) -> dict:
        community_mapping = {item.node: item.cluster for item in clusters}

        node_details: dict[str, dict] = {}
        for node in self.graph.nodes.values():
            if not isinstance(node, EntityNode):
                continue
            node_details[node.id] = {
                "name": node.name,
                "type": node.label,
                "description": node.properties.get("entity_description", ""),
            }

        community_info: dict[int, dict[str, list]] = {}
        for item in clusters:
            community_info.setdefault(item.cluster, {"entities": [], "relationships": []})
            if item.node in node_details:
                community_info[item.cluster]["entities"].append(node_details[item.node])

            for neighbor in nx_graph.neighbors(item.node):
                if community_mapping.get(neighbor) != item.cluster:
                    continue

                edge = nx_graph.get_edge_data(item.node, neighbor) or {}
                src_name = node_details.get(item.node, {}).get("name", item.node)
                tgt_name = node_details.get(neighbor, {}).get("name", neighbor)
                relationship = edge.get("relationship", "RELATED")
                description = edge.get("description", "")

                entry = f"{src_name} --[{relationship}]--> {tgt_name}"
                if description:
                    entry += f" ({description})"
                community_info[item.cluster]["relationships"].append(entry)

        return community_info

    def _generate_summaries(self, community_info: dict) -> None:
        llm = get_extraction_llm()
        self.community_summaries = {}
        for community_id, data in community_info.items():
            if not data["entities"] and not data["relationships"]:
                continue

            entities_text = "\n".join(
                f"- {item['name']} ({item['type']}): {item['description']}"
                for item in data["entities"]
                if item.get("name")
            )
            relationships_text = "\n".join(sorted(set(data["relationships"])))
            prompt = f"""You are analyzing a cluster of entities from news articles about
AI copyright, governance, and intellectual property.

Entities in this cluster:
{entities_text}

Relationships:
{relationships_text}

Write a concise briefing in 3 to 5 sentences that:
1. Identifies the main organizations, people, legal cases, or topics in this cluster
2. Explains how they are connected and why
3. Highlights any disputes, lawsuits, policy positions, or tensions
4. Notes what matters for understanding AI copyright or governance

Briefing:"""
            self.community_summaries[community_id] = llm.complete(prompt).text

    def get_community_summaries(self) -> dict:
        return self.community_summaries


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        summaries = self.graph_store.get_community_summaries()
        if not summaries:
            return "No community summaries found. Run the graph build step first."

        community_answers = [
            self._answer_from_community(summary, query_str)
            for summary in summaries.values()
        ]
        relevant_answers = [answer for answer in community_answers if answer.strip()]
        if not relevant_answers:
            return "I do not have enough information in the knowledge graph to answer that question."

        return self._aggregate(relevant_answers, query_str)

    def _answer_from_community(self, summary: str, query: str) -> str:
        prompt = (
            f"Community summary:\n{summary}\n\n"
            f"Question: {query}\n\n"
            "If this summary contains information relevant to the question, answer it. "
            "If not relevant, reply exactly: 'No relevant information.'\n\n"
            "Answer:"
        )
        response = get_extraction_llm().complete(prompt)
        text = response.text.strip()
        return "" if "no relevant information" in text.lower() else text

    def _aggregate(self, answers: list[str], query: str) -> str:
        combined = "\n\n---\n\n".join(answers)
        prompt = (
            "You have received answers from multiple knowledge graph communities.\n\n"
            f"Question: {query}\n\n"
            f"Community answers:\n{combined}\n\n"
            "Synthesize these into a single clear final answer. Remove redundancy, keep the important details, "
            "and answer the question directly.\n\n"
            "Final answer:"
        )
        return self.llm.complete(prompt).text


@dataclass(slots=True)
class PipelineResult:
    graph_store: GraphRAGStore
    query_engine: GraphRAGQueryEngine
    graph_json_path: Path
    graph_html_path: Path


def _build_documents(dataframe: pd.DataFrame) -> list[Document]:
    return [
        Document(
            text=str(row["full_text"]),
            metadata={
                "title": str(row.get("title", "")),
                "source": str(row.get("source", "")),
                "date": str(row.get("date", "")),
            },
        )
        for _, row in dataframe.iterrows()
        if str(row.get("full_text", "")).strip()
    ]


def run_pipeline(
    input_csv: Path,
    graph_json_path: Path = DEFAULT_GRAPH_JSON_PATH,
    graph_html_path: Path = DEFAULT_GRAPH_HTML_PATH,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    question: str | None = None,
    max_articles: int | None = None,
) -> PipelineResult:
    configure_models()

    dataframe = pd.read_csv(input_csv)
    if max_articles is not None:
        dataframe = dataframe.head(max_articles)

    print(f"Loaded {len(dataframe)} rows from {input_csv}")
    nodes = _build_documents(dataframe)
    graph_store = GraphRAGStore()
    extractor = GraphRAGExtractor(
        llm=get_extraction_llm(),
        extract_prompt=build_triplet_prompt(),
        max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
        num_workers=NUM_WORKERS,
    )

    PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[extractor],
        property_graph_store=graph_store,
        embed_kg_nodes=False,
        show_progress=True,
    )
    graph_store.build_communities()

    graph_data = export_graph_data(graph_store, graph_json_path)
    visualize_graph(graph_data, template_path, graph_html_path)

    query_engine = GraphRAGQueryEngine(
        graph_store=graph_store,
        llm=get_query_llm(),
    )

    if question:
        print()
        print("Question:")
        print(question)
        print()
        print("Answer:")
        print(query_engine.custom_query(question))

    return PipelineResult(
        graph_store=graph_store,
        query_engine=query_engine,
        graph_json_path=graph_json_path,
        graph_html_path=graph_html_path,
    )
