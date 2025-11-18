#!/usr/bin/env python3
"""
Test script for description utilities.
Creates dummy entities and tests the description functions.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List

from kvasir_ontology.entities.data_source.data_model import (
    DataSource, DataSourceBase, FileDataSourceBase
)
from kvasir_ontology.entities.dataset.data_model import (
    Dataset, DatasetBase, ObjectGroup, ObjectGroupBase,
    TimeSeriesGroupBase, TabularGroupBase, DataObject, DataObjectBase,
    TimeSeriesBase, TabularRowBase
)
from kvasir_ontology.entities.pipeline.data_model import (
    Pipeline, PipelineBase, PipelineImplementationBase, PipelineRunBase
)
from kvasir_ontology.entities.model.data_model import (
    ModelInstantiated, ModelInstantiatedBase, Model, ModelBase,
    ModelImplementation, ModelImplementationBase, ModelFunctionBase
)
from kvasir_ontology.entities.analysis.data_model import (
    Analysis, AnalysisBase, Section, AnalysisSectionBase,
    AnalysisCell, AnalysisCellBase, MarkdownCellBase, CodeCell, CodeCellBase,
    CodeOutput, CodeOutputBase
)
from kvasir_ontology.code.interface import CodeInterface
from kvasir_ontology.code.data_model import CodebaseFile, CodebasePath
from kvasir_ontology.graph.interface import GraphInterface
from kvasir_ontology.graph.data_model import EdgeDefinition
from kvasir_ontology.ontology import Ontology
from kvasir_ontology._description_utils import (
    get_data_source_description,
    get_dataset_description,
    get_pipeline_description,
    get_pipeline_run_description,
    get_model_entity_description,
    get_analysis_description
)


class DummyCodeInterface(CodeInterface):
    """Dummy implementation of CodeInterface for testing."""

    def __init__(self, user_id: uuid.UUID, mount_group_id: uuid.UUID):
        super().__init__(user_id, mount_group_id)
        self._files = {
            "src/pipelines/cleaning.py": """def clean_data(data_path: str) -> dict:
    \"\"\"Clean the raw data file.\"\"\"
    import pandas as pd
    df = pd.read_csv(data_path)
    df = df.dropna()
    return {"cleaned_data": df}
""",
            "src/models/forecasting_model.py": """import torch
import torch.nn as nn

class ForecastingModel(nn.Module):
    \"\"\"A simple forecasting model.\"\"\"
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
"""
        }

    async def get_codebase_tree(self, mount_group_id: uuid.UUID) -> CodebasePath:
        return CodebasePath(
            path="/",
            is_file=False,
            sub_paths=[]
        )

    async def get_codebase_file(self, file_path: str) -> CodebaseFile:
        if file_path in self._files:
            return CodebaseFile(
                path=file_path,
                content=self._files[file_path]
            )
        raise FileNotFoundError(f"File not found: {file_path}")


class DummyGraphInterface(GraphInterface):
    """Dummy implementation of GraphInterface for testing."""

    def __init__(self, user_id: uuid.UUID, edges_map: dict):
        # Don't call super().__init__() because it tries to instantiate abstract interfaces
        self.user_id = user_id
        self.edges_map = edges_map  # Maps entity_id -> List[EdgeDefinition]
        # Set dummy interfaces to avoid attribute errors
        self.data_sources = None
        self.datasets = None
        self.pipelines = None
        self.models = None
        self.analyses = None

    async def get_node_edges(self, node_id: uuid.UUID) -> List[EdgeDefinition]:
        """Return edges for a given node."""
        return self.edges_map.get(node_id, [])

    # Stub implementations for other required methods
    async def add_node(self, node):
        pass

    async def get_node(self, node_id: uuid.UUID):
        pass

    async def delete_node(self, node_id: uuid.UUID) -> None:
        pass

    async def get_node_groups(self, node_id: uuid.UUID):
        return []

    async def create_node_group(self, node_group):
        pass

    async def delete_node_group(self, node_group_id: uuid.UUID) -> None:
        pass

    async def add_node_to_group(self, node_id: uuid.UUID, node_group_id: uuid.UUID) -> None:
        pass

    async def remove_nodes_from_groups(self, node_ids: List[uuid.UUID], node_group_ids: List[uuid.UUID]) -> None:
        pass

    async def create_edges(self, edges: List[EdgeDefinition]) -> None:
        pass

    async def remove_edges(self, edges: List[EdgeDefinition]) -> None:
        pass

    async def get_entity_graph(self, root_group_id=None, root_node_id=None):
        pass


def create_dummy_data_source() -> DataSource:
    """Create a dummy DataSource."""
    now = datetime.now(timezone.utc)
    return DataSource(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        type="file",
        name="sales_data.csv",
        description="Monthly sales data for Q1 2024. Contains product sales, revenue, and customer information.",
        additional_variables={
            "source": "CRM system",
            "export_date": "2024-01-15",
            "notes": "Data has been validated and cleaned"
        },
        created_at=now,
        type_fields=FileDataSourceBase(
            id=uuid.uuid4(),
            file_name="sales_data.csv",
            file_path="/data/raw/sales_data.csv",
            file_type="csv",
            file_size_bytes=245678,
            created_at=now,
            updated_at=now
        )
    )


def create_dummy_dataset() -> Dataset:
    """Create a dummy Dataset."""
    now = datetime.now(timezone.utc)

    # Create a time series group
    time_series_group = TimeSeriesGroupBase(
        id=uuid.uuid4(),
        total_timestamps=1000,
        number_of_series=50,
        sampling_frequency="d",
        timezone="UTC",
        features_schema={"temperature": "float",
                         "humidity": "float", "pressure": "float"},
        earliest_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        latest_timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc),
        created_at=now,
        updated_at=now
    )

    # Create a data object for the group
    data_object = DataObject(
        id=uuid.uuid4(),
        name="weather_series_001",
        group_id=uuid.uuid4(),
        original_id="ws_001",
        description="Weather time series for station 001",
        additional_variables={"station_id": "001", "location": "New York"},
        created_at=now,
        updated_at=now,
        modality_fields=TimeSeriesBase(
            id=uuid.uuid4(),
            start_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_timestamp=datetime(2024, 3, 31, tzinfo=timezone.utc),
            num_timestamps=90,
            sampling_frequency="d",
            timezone="UTC",
            features_schema={"temperature": "float", "humidity": "float"},
            created_at=now,
            updated_at=now
        )
    )

    # Create object group
    object_group = ObjectGroup(
        id=uuid.uuid4(),
        name="Weather Time Series",
        description="Daily weather measurements from multiple stations",
        modality="time_series",
        dataset_id=uuid.uuid4(),
        original_id_name="station_id",
        additional_variables={"region": "Northeast",
                              "collection_method": "automated"},
        echart_id=None,
        created_at=now,
        updated_at=now,
        modality_fields=time_series_group,
        first_data_object=data_object
    )

    return Dataset(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        name="Weather Dataset Q1 2024",
        description="Comprehensive weather dataset containing daily measurements from 50 weather stations across the Northeast region for Q1 2024.",
        additional_variables={
            "region": "Northeast",
            "quality_score": 0.95,
            "last_updated": "2024-04-01"
        },
        created_at=now,
        updated_at=now,
        object_groups=[object_group]
    )


def create_dummy_pipeline() -> Pipeline:
    """Create a dummy Pipeline."""
    now = datetime.now(timezone.utc)

    implementation = PipelineImplementationBase(
        id=uuid.uuid4(),
        python_function_name="clean_data",
        docstring="Clean raw data by removing nulls and outliers.",
        description="This pipeline processes raw CSV files, removes null values, filters outliers, and returns cleaned data.",
        args_schema={"data_path": "str", "output_path": "str"},
        default_args={"output_path": "/data/cleaned/"},
        output_variables_schema={"cleaned_data": "DataFrame", "stats": "dict"},
        implementation_script_path="src/pipelines/cleaning.py",
        created_at=now,
        updated_at=now
    )

    return Pipeline(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        name="Data Cleaning Pipeline",
        description="Pipeline for cleaning and preprocessing raw data files. Handles missing values, outliers, and data type conversions.",
        created_at=now,
        updated_at=now,
        implementation=implementation,
        runs=[]
    )


def create_dummy_pipeline_runs(pipeline_id: uuid.UUID) -> List[PipelineRunBase]:
    """Create dummy pipeline runs."""
    now = datetime.now(timezone.utc)

    run1 = PipelineRunBase(
        id=uuid.uuid4(),
        pipeline_id=pipeline_id,
        args={"data_path": "/data/raw/weather.csv",
              "output_path": "/data/cleaned/"},
        output_variables={"cleaned_data": "DataFrame",
                          "stats": {"rows": 1000, "columns": 10}},
        name="Run 1 - Weather Data Cleaning",
        description="First run of the cleaning pipeline on weather data",
        status="completed",
        start_time=now,
        end_time=datetime.now(timezone.utc),
        created_at=now,
        updated_at=now
    )

    run2 = PipelineRunBase(
        id=uuid.uuid4(),
        pipeline_id=pipeline_id,
        args={"data_path": "/data/raw/sales.csv",
              "output_path": "/data/cleaned/"},
        output_variables={"cleaned_data": "DataFrame",
                          "stats": {"rows": 500, "columns": 8}},
        name="Run 2 - Sales Data Cleaning",
        description="Second run of the cleaning pipeline on sales data",
        status="completed",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        created_at=now,
        updated_at=now
    )

    return [run1, run2]


def create_dummy_model_instantiated() -> ModelInstantiated:
    """Create a dummy ModelInstantiated."""
    now = datetime.now(timezone.utc)

    # Create training function
    training_function = ModelFunctionBase(
        id=uuid.uuid4(),
        docstring="Train the forecasting model on time series data.",
        args_schema={"data": "DataFrame",
                     "epochs": "int", "learning_rate": "float"},
        default_args={"epochs": 100, "learning_rate": 0.001},
        output_variables_schema={
            "model_weights": "dict", "training_loss": "list"},
        created_at=now,
        updated_at=now
    )

    # Create inference function
    inference_function = ModelFunctionBase(
        id=uuid.uuid4(),
        docstring="Generate forecasts using the trained model.",
        args_schema={"model_weights": "dict",
                     "input_data": "DataFrame", "horizon": "int"},
        default_args={"horizon": 7},
        output_variables_schema={
            "forecasts": "DataFrame", "confidence_intervals": "DataFrame"},
        created_at=now,
        updated_at=now
    )

    # Create model implementation
    model_implementation = ModelImplementation(
        id=uuid.uuid4(),
        modality="time_series",
        task="forecasting",
        public=False,
        python_class_name="ForecastingModel",
        description="LSTM-based forecasting model for time series data. Supports multi-step ahead forecasting with confidence intervals.",
        user_id=uuid.uuid4(),
        source="github",
        training_function_id=training_function.id,
        inference_function_id=inference_function.id,
        implementation_script_path="src/models/forecasting_model.py",
        model_class_docstring="A simple forecasting model using LSTM architecture.",
        default_config={"input_size": 10, "hidden_size": 64},
        config_schema={"input_size": "int", "hidden_size": "int"},
        created_at=now,
        updated_at=now,
        training_function=training_function,
        inference_function=inference_function
    )

    # Create model
    model = Model(
        id=uuid.uuid4(),
        name="Time Series Forecasting Model",
        user_id=uuid.uuid4(),
        description="LSTM-based model for time series forecasting",
        created_at=now,
        updated_at=now,
        implementation=model_implementation
    )

    # Create model instantiated
    return ModelInstantiated(
        id=uuid.uuid4(),
        model_id=model.id,
        name="Weather Forecast Model v1",
        user_id=uuid.uuid4(),
        description="Instantiated forecasting model trained on weather data. Configured with 10 input features and 64 hidden units.",
        config={"input_size": 10, "hidden_size": 64, "dropout": 0.2},
        weights_save_dir="/models/weather_forecast_v1/weights",
        created_at=now,
        updated_at=now,
        model=model
    )


def create_dummy_analysis() -> Analysis:
    """Create a dummy Analysis."""
    now = datetime.now(timezone.utc)

    # Create markdown cell
    markdown_cell = AnalysisCell(
        id=uuid.uuid4(),
        order=0,
        type="markdown",
        section_id=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        type_fields=MarkdownCellBase(
            id=uuid.uuid4(),
            markdown="# Weather Analysis\n\nThis analysis explores temperature trends in Q1 2024.",
            created_at=now,
            updated_at=now
        )
    )

    # Create code cell with output
    code_output = CodeOutput(
        id=uuid.uuid4(),
        code_cell_id=uuid.uuid4(),
        output="Mean temperature: 45.2°F\nMax temperature: 72.1°F\nMin temperature: 18.5°F",
        created_at=now,
        updated_at=now,
        images=[],
        echarts=[],
        tables=[]
    )

    code_cell = AnalysisCell(
        id=uuid.uuid4(),
        order=1,
        type="code",
        section_id=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        type_fields=CodeCell(
            id=uuid.uuid4(),
            code="import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('weather_data.csv')\nprint(f'Mean temperature: {df[\"temperature\"].mean():.1f}°F')\nprint(f'Max temperature: {df[\"temperature\"].max():.1f}°F')\nprint(f'Min temperature: {df[\"temperature\"].min():.1f}°F')",
            created_at=now,
            updated_at=now,
            output=code_output
        )
    )

    # Create section
    section = Section(
        id=uuid.uuid4(),
        name="Temperature Analysis",
        analysis_id=uuid.uuid4(),
        description="Analysis of temperature trends and statistics",
        created_at=now,
        updated_at=now,
        cells=[markdown_cell, code_cell]
    )

    return Analysis(
        id=uuid.uuid4(),
        name="Weather Data Analysis Q1 2024",
        description="Comprehensive analysis of weather patterns and trends for the first quarter of 2024. Includes temperature analysis, humidity patterns, and precipitation statistics.",
        created_at=now,
        updated_at=now,
        sections=[section]
    )


async def main():
    """Run all description tests."""
    print("=" * 80)
    print("TESTING DESCRIPTION UTILITIES")
    print("=" * 80)
    print()

    # Create dummy entities first
    data_source = create_dummy_data_source()
    dataset = create_dummy_dataset()
    pipeline = create_dummy_pipeline()
    # Add pipeline runs to the pipeline
    pipeline_runs = create_dummy_pipeline_runs(pipeline.id)
    pipeline.runs = pipeline_runs
    model_entity = create_dummy_model_instantiated()
    analysis = create_dummy_analysis()

    # Create dummy code interface
    user_id = uuid.uuid4()
    mount_group_id = uuid.uuid4()
    code_interface = DummyCodeInterface(user_id, mount_group_id)

    # Create dummy entity interfaces that return our entities
    class MockDataSourceInterface:
        async def get_data_sources(self, ids: List[uuid.UUID]):
            return [data_source] if data_source.id in ids else []

    class MockDatasetInterface:
        async def get_datasets(self, ids: List[uuid.UUID]):
            return [dataset] if dataset.id in ids else []

    class MockPipelineInterface:
        async def get_pipelines(self, ids: List[uuid.UUID]):
            return [pipeline] if pipeline.id in ids else []

        async def get_pipeline(self, pipeline_id: uuid.UUID):
            return pipeline if pipeline.id == pipeline_id else None

        async def get_pipeline_run(self, run_id: uuid.UUID):
            for run in pipeline_runs:
                if run.id == run_id:
                    return run
            return None

    class MockModelInterface:
        async def get_models_instantiated(self, ids: List[uuid.UUID]):
            return [model_entity] if model_entity.id in ids else []

    class MockAnalysisInterface:
        async def get_analyses(self, ids: List[uuid.UUID]):
            return [analysis] if analysis.id in ids else []

    class MockPipelineRunInterface:
        async def get_pipeline_runs(self, ids: List[uuid.UUID]):
            return [run for run in pipeline_runs if run.id in ids]

    # Create edges map with valid edge types:
    # - data_source -> dataset
    # - data_source -> pipeline
    # - data_source -> analysis
    # - dataset -> pipeline
    # - dataset -> analysis
    # - model_instantiated -> pipeline
    # - model_instantiated -> analysis
    edges_map = {
        # Dataset has input from data_source
        dataset.id: [
            EdgeDefinition(
                from_node_type="data_source",
                from_node_id=data_source.id,
                to_node_type="dataset",
                to_node_id=dataset.id
            )
        ],
        # Pipeline has inputs from dataset and model_entity
        pipeline.id: [
            EdgeDefinition(
                from_node_type="dataset",
                from_node_id=dataset.id,
                to_node_type="pipeline",
                to_node_id=pipeline.id
            ),
            EdgeDefinition(
                from_node_type="model_instantiated",
                from_node_id=model_entity.id,
                to_node_type="pipeline",
                to_node_id=pipeline.id
            )
        ],
        # Analysis has inputs from dataset and data_source
        analysis.id: [
            EdgeDefinition(
                from_node_type="dataset",
                from_node_id=dataset.id,
                to_node_type="analysis",
                to_node_id=analysis.id
            ),
            EdgeDefinition(
                from_node_type="data_source",
                from_node_id=data_source.id,
                to_node_type="analysis",
                to_node_id=analysis.id
            )
        ],
        # Model entity has output to pipeline
        model_entity.id: [
            EdgeDefinition(
                from_node_type="model_instantiated",
                from_node_id=model_entity.id,
                to_node_type="pipeline",
                to_node_id=pipeline.id
            )
        ],
        # Data source has outputs to dataset and analysis
        data_source.id: [
            EdgeDefinition(
                from_node_type="data_source",
                from_node_id=data_source.id,
                to_node_type="dataset",
                to_node_id=dataset.id
            ),
            EdgeDefinition(
                from_node_type="data_source",
                from_node_id=data_source.id,
                to_node_type="analysis",
                to_node_id=analysis.id
            )
        ],
        # Pipeline run 1 has input from dataset and output to dataset
        pipeline_runs[0].id: [
            EdgeDefinition(
                from_node_type="dataset",
                from_node_id=dataset.id,
                to_node_type="pipeline_run",
                to_node_id=pipeline_runs[0].id
            ),
            EdgeDefinition(
                from_node_type="pipeline_run",
                from_node_id=pipeline_runs[0].id,
                to_node_type="dataset",
                to_node_id=dataset.id
            )
        ],
        # Pipeline run 2 has input from dataset
        pipeline_runs[1].id: [
            EdgeDefinition(
                from_node_type="dataset",
                from_node_id=dataset.id,
                to_node_type="pipeline_run",
                to_node_id=pipeline_runs[1].id
            )
        ]
    }

    graph_interface = DummyGraphInterface(user_id, edges_map)

    ontology = Ontology(
        user_id=user_id,
        mount_group_id=uuid.uuid4(),
        data_source_interface=MockDataSourceInterface(),
        analysis_interface=MockAnalysisInterface(),
        dataset_interface=MockDatasetInterface(),
        pipeline_interface=MockPipelineInterface(),
        model_interface=MockModelInterface(),
        visualization_interface=type('MockInterface', (), {})(),
        graph_interface=graph_interface,
        code_interface=code_interface
    )

    # Test DataSource
    print("\n" + "=" * 80)
    print("DATA SOURCE DESCRIPTION")
    print("=" * 80)
    print(await get_data_source_description(data_source, ontology))

    # Test Dataset
    print("\n" + "=" * 80)
    print("DATASET DESCRIPTION")
    print("=" * 80)
    print(await get_dataset_description(dataset, ontology))

    # Test Pipeline
    print("\n" + "=" * 80)
    print("PIPELINE DESCRIPTION")
    print("=" * 80)
    print(await get_pipeline_description(pipeline, ontology))

    # Test Model Instantiated
    print("\n" + "=" * 80)
    print("MODEL ENTITY DESCRIPTION")
    print("=" * 80)
    print(await get_model_entity_description(model_entity, ontology))

    # Test Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS DESCRIPTION")
    print("=" * 80)
    print(await get_analysis_description(analysis, ontology))

    # Test Pipeline with Runs (should show runs nested)
    print("\n" + "=" * 80)
    print("PIPELINE DESCRIPTION WITH RUNS")
    print("=" * 80)
    print(await get_pipeline_description(pipeline, ontology, include_runs=True))

    # Test Pipeline Run directly (should show inputs/outputs and pipeline description)
    print("\n" + "=" * 80)
    print("PIPELINE RUN DESCRIPTION (DIRECT)")
    print("=" * 80)
    print(await get_pipeline_run_description(pipeline_runs[0], pipeline, ontology, show_pipeline_description=True, include_connections=True))

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
