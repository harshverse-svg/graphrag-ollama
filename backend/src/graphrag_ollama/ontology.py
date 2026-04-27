from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ENTITY_TYPES = [
    "ORGANIZATION",
    "PERSON",
    "LEGISLATION",
    "LEGAL_CASE",
    "CONCEPT",
    "GOVERNMENT",
    "AI_SYSTEM",
]

RELATION_TYPES = [
    "FILED_AGAINST",
    "DEFENDANT_IN",
    "REGULATES",
    "ADVOCATES_FOR",
    "TRAINED_ON",
    "PART_OF",
    "REFERENCES",
    "OPPOSES",
]

EntityTypeStr = Literal[
    "ORGANIZATION",
    "PERSON",
    "LEGISLATION",
    "LEGAL_CASE",
    "CONCEPT",
    "GOVERNMENT",
    "AI_SYSTEM",
]

RelationTypeStr = Literal[
    "FILED_AGAINST",
    "DEFENDANT_IN",
    "REGULATES",
    "ADVOCATES_FOR",
    "TRAINED_ON",
    "PART_OF",
    "REFERENCES",
    "OPPOSES",
]


def build_triplet_prompt() -> str:
    entity_types_str = ", ".join(ENTITY_TYPES)
    relation_types_str = ", ".join(RELATION_TYPES)
    return f"""
-Goal-
Given a news article about AI copyright, governance, or intellectual property,
identify all entities mentioned in the article and their relationships.

Extract up to {{max_knowledge_triplets}} entity-relation triplets.

-Allowed Entity Types-
{entity_types_str}

-Allowed Relationship Types-
{relation_types_str}

-Steps-
1. Identify all relevant entities. For each entity extract:
   - name: Name of the entity, capitalized
   - type: One of the allowed entity types above
   - description: A brief description of the entity and its role in AI copyright/governance

2. Identify relationships between entities. For each pair extract:
   - source: name of the source entity
   - target: name of the target entity
   - relation: one of the allowed relationship types above
   - description: a sentence explaining why and how these entities are related

-Real Data-
######################
text: {{text}}
######################
"""


class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of the entity, capitalized")
    type: EntityTypeStr = Field(description="One of the allowed entity types")
    description: str = Field(description="Brief description of the entity and its role")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation: RelationTypeStr = Field(description="One of the allowed relationship types")
    description: str = Field(description="Sentence explaining the relationship")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
