import yaml
from uuid import UUID
from typing import List, Literal, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, model_validator

NODE_TYPE_LITERAL = Literal["data_source", "dataset", "analysis",
                            "pipeline", "model_instantiated", "pipeline_run"]


class EntityNodeBase(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    entity_type: NODE_TYPE_LITERAL
    x_position: float
    y_position: float
    created_at: datetime
    updated_at: datetime


# The node group is quite flexible and functions like a mountpoint into the entity graph
# You can think of it as a project or a folder in a file (entity) system
# A group can correspond to a Python package (projects will)
class NodeGroupBase(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    python_package_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class NodeInGroup(BaseModel):
    node_id: UUID
    node_group_id: UUID
    created_at: datetime
    updated_at: datetime


class DatasetFromDataSource(BaseModel):
    data_source_id: UUID
    dataset_id: UUID
    created_at: datetime
    updated_at: datetime


class DataSourceSupportedInPipeline(BaseModel):
    data_source_id: UUID
    pipeline_id: UUID
    created_at: datetime
    updated_at: datetime


class DatasetSupportedInPipeline(BaseModel):
    dataset_id: UUID
    pipeline_id: UUID
    created_at: datetime
    updated_at: datetime


class ModelEntitySupportedInPipeline(BaseModel):
    model_instantiated_id: UUID
    pipeline_id: UUID
    created_at: datetime
    updated_at: datetime


class DatasetInPipelineRun(BaseModel):
    pipeline_run_id: UUID
    dataset_id: UUID
    created_at: datetime
    updated_at: datetime


class DataSourceInPipelineRun(BaseModel):
    pipeline_run_id: UUID
    data_source_id: UUID
    created_at: datetime
    updated_at: datetime


class ModelEntityInPipelineRun(BaseModel):
    pipeline_run_id: UUID
    model_instantiated_id: UUID
    created_at: datetime
    updated_at: datetime


class PipelineRunOutputDataset(BaseModel):
    pipeline_run_id: UUID
    dataset_id: UUID
    created_at: datetime
    updated_at: datetime


class PipelineRunOutputModelEntity(BaseModel):
    pipeline_run_id: UUID
    model_instantiated_id: UUID
    created_at: datetime
    updated_at: datetime


class PipelineRunOutputDataSource(BaseModel):
    pipeline_run_id: UUID
    data_source_id: UUID
    created_at: datetime
    updated_at: datetime


class DataSourceInAnalysis(BaseModel):
    analysis_id: UUID
    data_source_id: UUID
    created_at: datetime
    updated_at: datetime


class DatasetInAnalysis(BaseModel):
    analysis_id: UUID
    dataset_id: UUID
    created_at: datetime
    updated_at: datetime


class ModelInstantiatedInAnalysis(BaseModel):
    analysis_id: UUID
    model_instantiated_id: UUID
    created_at: datetime
    updated_at: datetime


###

class EdgePoints(BaseModel):
    data_sources: List[UUID] = []
    datasets: List[UUID] = []
    analyses: List[UUID] = []
    pipelines: List[UUID] = []
    models_instantiated: List[UUID] = []
    pipeline_runs: List[UUID] = []


class EntityNode(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    x_position: float
    y_position: float
    from_entities: EdgePoints
    to_entities: EdgePoints


# This one is different, outputs go through runs
class PipelineNode(EntityNode):
    id: UUID
    description: Optional[str] = None
    x_position: float
    y_position: float
    from_entities: EdgePoints
    runs: List[EntityNode] = []


class EntityGraph(BaseModel):
    data_sources: List[EntityNode] = []
    datasets: List[EntityNode] = []
    pipelines: List[PipelineNode] = []
    analyses: List[EntityNode] = []
    models_instantiated: List[EntityNode] = []


###


class NodeGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    python_package_name: Optional[str] = None


class EntityNodeCreate(BaseModel):
    # Must foreign key to an entity (data source, dataset, analysis, pipeline, model entity)
    id: UUID
    name: str
    entity_type: NODE_TYPE_LITERAL
    x_position: float
    y_position: float
    description: Optional[str] = None
    node_groups: Optional[List[UUID]] = None


# Valid edge types for entity graph
VALID_EDGE_TYPES: List[Tuple[str, str]] = [
    ("data_source", "dataset"),
    ("data_source", "pipeline"),
    ("data_source", "analysis"),
    ("dataset", "pipeline"),
    ("dataset", "analysis"),
    ("model_instantiated", "pipeline"),
    ("model_instantiated", "analysis"),
]

# Valid edge types involving pipeline runs
PIPELINE_RUN_EDGE_TYPES: List[Tuple[str, str]] = [
    ("dataset", "pipeline_run"),
    ("data_source", "pipeline_run"),
    ("model_instantiated", "pipeline_run"),
    ("pipeline_run", "dataset"),
    ("pipeline_run", "model_instantiated"),
    ("pipeline_run", "data_source"),
]


class EdgeDefinition(BaseModel):
    from_node_type: NODE_TYPE_LITERAL
    from_node_id: UUID
    to_node_type: NODE_TYPE_LITERAL
    to_node_id: UUID

    @model_validator(mode='after')
    def validate_edge_type(self) -> 'EdgeDefinition':
        """Validate that this edge uses a valid node type combination."""
        all_valid_edges = VALID_EDGE_TYPES + PIPELINE_RUN_EDGE_TYPES
        edge_type = (self.from_node_type, self.to_node_type)

        if edge_type not in all_valid_edges:
            valid_edges_str = "\n".join(
                [f"  - {from_type} -> {to_type}" for from_type, to_type in all_valid_edges])
            raise ValueError(
                f"Invalid edge type: {self.from_node_type} -> {self.to_node_type}\n\n"
                f"Valid edge types:\n{valid_edges_str}"
            )

        return self


class EdgeDefinitionUsingNames(BaseModel):
    from_node_type: NODE_TYPE_LITERAL
    from_node_name: str
    to_node_type: NODE_TYPE_LITERAL
    to_node_name: str

    @model_validator(mode='after')
    def validate_edge_type(self) -> 'EdgeDefinitionUsingNames':
        """Validate that this edge uses a valid node type combination."""
        all_valid_edges = VALID_EDGE_TYPES + PIPELINE_RUN_EDGE_TYPES
        edge_type = (self.from_node_type, self.to_node_type)

        if edge_type not in all_valid_edges:
            valid_edges_str = "\n".join(
                [f"  - {from_type} -> {to_type}" for from_type, to_type in all_valid_edges])
            raise ValueError(
                f"Invalid edge type: {self.from_node_type} -> {self.to_node_type}\n\n"
                f"Valid edge types:\n{valid_edges_str}"
            )

        return self


def get_entity_graph_description(entity_graph: EntityGraph) -> str:
    graph_dict = entity_graph.model_dump(mode="json")

    # Build mapping of UUID to {name}_uuid_{ID} format
    id_to_readable_map = {}

    # Collect all entities with their names and IDs
    def _collect_entities(entities: List[dict], entity_type: str):
        for entity in entities:
            if 'id' in entity and 'name' in entity:
                entity_id = entity['id']
                entity_name = entity['name'].lower().replace(
                    ' ', '_').replace('-', '_')
                # Remove special characters
                entity_name = ''.join(
                    c for c in entity_name if c.isalnum() or c == '_')
                readable_id = f"{entity_name}_UUID_{entity_id}"
                id_to_readable_map[entity_id] = readable_id

            # Handle nested runs in pipelines
            if 'runs' in entity:
                _collect_entities(entity['runs'], 'pipeline_run')

    # Collect from all entity types
    if 'data_sources' in graph_dict:
        _collect_entities(graph_dict['data_sources'], 'data_source')
    if 'datasets' in graph_dict:
        _collect_entities(graph_dict['datasets'], 'dataset')
    if 'pipelines' in graph_dict:
        _collect_entities(graph_dict['pipelines'], 'pipeline')
    if 'analyses' in graph_dict:
        _collect_entities(graph_dict['analyses'], 'analysis')
    if 'models_instantiated' in graph_dict:
        _collect_entities(
            graph_dict['models_instantiated'], 'model_instantiated')
    if 'pipeline_runs' in graph_dict:
        _collect_entities(graph_dict['pipeline_runs'], 'pipeline_run')

    # Replace IDs with readable format
    def _replace_ids(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k == 'id' and isinstance(v, str) and v in id_to_readable_map:
                    result[k] = id_to_readable_map[v]
                else:
                    result[k] = _replace_ids(v)
            return result
        elif isinstance(value, list):
            return [id_to_readable_map.get(item, _replace_ids(item)) if isinstance(item, str) else _replace_ids(item) for item in value]
        return value

    graph_dict = _replace_ids(graph_dict)

    def _remove_empty_fields(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {k: _remove_empty_fields(v) for k, v in value.items()}
            return {k: v for k, v in cleaned.items() if v is not None and v != {} and v != []}
        elif isinstance(value, list):
            cleaned = [_remove_empty_fields(item) for item in value]
            return [item for item in cleaned if item is not None and item != {} and item != []]
        elif not value and value != 0 and value is not False:
            return None
        return value

    graph_dict = _remove_empty_fields(graph_dict)

    # Generate YAML
    yaml_content = yaml.dump(
        graph_dict, sort_keys=False, default_flow_style=False)

    annotations = []
    annotations.append("# Entity Graph Representation")
    annotations.append("#")
    annotations.append(
        "# NOTE: We represent entity IDs in the format {name}_UUID_{ID}.")
    annotations.append(
        "# When submitting data related to a specific entity, use just the ID part after 'UUID_'.")
    annotations.append("#")
    annotations.append("# List of entities in this graph:")

    for readable_id in sorted(id_to_readable_map.values()):
        annotations.append(f"#   - {readable_id}")

    annotations.append("#")
    annotations.append("")

    # Combine annotations and YAML
    return "\n".join(annotations) + yaml_content
