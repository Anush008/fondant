import datetime
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.pipeline import ComponentOp, Pipeline, Resources
from fondant.pipeline.compiler import (
    DockerCompiler,
    KubeFlowCompiler,
    VertexCompiler,
)
from fondant.testing import (
    DockerPipelineConfigs,
    KubeflowPipelineConfigs,
    VertexPipelineConfigs,
)

COMPONENTS_PATH = Path("./tests/example_pipelines/valid_pipeline")

VALID_PIPELINE = Path("./tests/example_pipelines/compiled_pipeline/")

TEST_PIPELINES = [
    (
        "example_1",
        [
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows=10,
                    resources=Resources(
                        memory_limit="512M",
                        memory_request="256M",
                    ),
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "second_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows=10,
                ),
                "cache_key": "2",
            },
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "third_component"),
                    arguments={
                        "storage_args": "a dummy string arg",
                    },
                ),
                "cache_key": "3",
            },
        ],
    ),
    (
        "example_2",
        [
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp.from_registry(
                    name="image_cropping",
                    arguments={"cropping_threshold": 0, "padding": 0},
                ),
                "cache_key": "2",
            },
        ],
    ),
]


@pytest.fixture()
def _freeze_time(monkeypatch):
    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls):
            return datetime.datetime(2023, 1, 1)

    monkeypatch.setattr(
        datetime,
        "datetime",
        FrozenDatetime,
    )


@pytest.fixture(params=TEST_PIPELINES)
def setup_pipeline(request, tmp_path, monkeypatch):
    pipeline = Pipeline(
        pipeline_name="testpipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    example_dir, components = request.param
    prev_comp = None
    cache_dict = {}
    for component_dict in components:
        component = component_dict["component_op"]
        cache_key = component_dict["cache_key"]
        # set the cache_key as a default argument in the lambda function to avoid setting attribute
        # by reference
        monkeypatch.setattr(
            component,
            "get_component_cache_key",
            lambda cache_key=cache_key, previous_component_cache=None: cache_key,
        )
        pipeline.add_op(component, dependencies=prev_comp)
        prev_comp = component
        cache_dict[component.name] = cache_key

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    return example_dir, pipeline, cache_dict


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, pipeline, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path, build_args=[])
        pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.pipeline_name == pipeline.name
        assert pipeline_configs.pipeline_description == pipeline.description
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get exepcted component configs
            component = pipeline._graph[component_name]
            component_op = component["fondant_component_op"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_local_path(setup_pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only created for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, cache_dict = setup_pipeline
        work_dir = f"/{fn.stem}"
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
        expected_run_id = "testpipeline-20230101000000"
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # check if volumes are defined correctly

            cache_key = cache_dict[component_name]
            assert component_configs.volumes == [
                {
                    "source": str(fn),
                    "target": work_dir,
                    "type": "bind",
                },
            ]
            cleaned_pipeline_name = pipeline.name.replace("_", "")
            # check if commands are patched to use the working dir
            expected_output_manifest_path = (
                f"{work_dir}/{cleaned_pipeline_name}/{expected_run_id}"
                f"/{component_name}/manifest.json"
            )
            expected_metadata = (
                f'{{"base_path": "{work_dir}", "pipeline_name": '
                f'"{cleaned_pipeline_name}", "run_id": "{expected_run_id}", '
                f'"component_id": "{component_name}", "cache_key": "{cache_key}"}}'
            )

            assert (
                component_configs.arguments["output_manifest_path"]
                == expected_output_manifest_path
            )
            assert component_configs.arguments["metadata"] == expected_metadata


@pytest.mark.usefixtures("_freeze_time")
def test_docker_remote_path(setup_pipeline, tmp_path_factory):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    _, pipeline, cache_dict = setup_pipeline
    remote_dir = "gs://somebucket/artifacts"
    pipeline.base_path = remote_dir
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
        expected_run_id = "testpipeline-20230101000000"
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            cache_key = cache_dict[component_name]
            # check that no volumes are created
            assert component_configs.volumes == []
            # check if commands are patched to use the remote dir
            cleaned_pipeline_name = pipeline.name.replace("_", "")

            expected_output_manifest_path = (
                f"{remote_dir}/{cleaned_pipeline_name}/{expected_run_id}"
                f"/{component_name}/manifest.json"
            )

            expected_metadata = (
                f'{{"base_path": "{remote_dir}", "pipeline_name": '
                f'"{cleaned_pipeline_name}", "run_id": "{expected_run_id}", '
                f'"component_id": "{component_name}", "cache_key": "{cache_key}"}}'
            )

            assert (
                component_configs.arguments["output_manifest_path"]
                == expected_output_manifest_path
            )
            assert component_configs.arguments["metadata"] == expected_metadata


@pytest.mark.usefixtures("_freeze_time")
def test_docker_extra_volumes(setup_pipeline, tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, _ = setup_pipeline
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        # define some extra volumes to be mounted
        extra_volumes = ["hello:there", "general:kenobi"]
        output_path = str(fn / "docker-compose.yml")

        compiler.compile(
            pipeline=pipeline,
            output_path=output_path,
            extra_volumes=extra_volumes,
        )

        pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
        for _, service in pipeline_configs.component_configs.items():
            assert all(
                extra_volume in service.volumes for extra_volume in extra_volumes
            )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_configuration(tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="GPU",
        ),
    )

    pipeline.add_op(component_1)
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yaml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
        component_config = pipeline_configs.component_configs["first_component"]
        assert component_config.accelerators[0].type == "gpu"
        assert component_config.accelerators[0].number == 1


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_docker_configuration(tmp_path_factory):
    """Test that a valid error is returned when an unknown accelerator is set."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
    )

    pipeline.add_op(component_1)
    compiler = DockerCompiler()
    with pytest.raises(InvalidPipelineDefinition):
        compiler.compile(pipeline=pipeline, output_path="kubeflow_pipeline.yml")


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to kubeflow."""
    example_dir, pipeline, _ = setup_pipeline
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.pipeline_name == pipeline.name
        assert pipeline_configs.pipeline_description == pipeline.description
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get exepcted component configs
            component = pipeline._graph[component_name]
            component_op = component["fondant_component_op"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_configuration(tmp_path_factory):
    """Test that the kubeflow pipeline can be configured."""
    node_pool_label = "dummy_label"
    node_pool_name = "dummy_label"

    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            node_pool_label=node_pool_label,
            node_pool_name=node_pool_name,
            accelerator_number=1,
            accelerator_name="GPU",
        ),
    )
    pipeline.add_op(component_1)
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)
        component_configs = pipeline_configs.component_configs["first_component"]
        for accelerator in component_configs.accelerators:
            assert accelerator.type == "nvidia.com/gpu"
            assert accelerator.number == 1
        assert component_configs.node_pool_label == node_pool_label
        assert component_configs.node_pool_name == node_pool_name


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_kubeflow_configuration(tmp_path_factory):
    """Test that an error is returned when an invalid resource is provided."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
    )

    pipeline.add_op(component_1)
    compiler = KubeFlowCompiler()
    with pytest.raises(InvalidPipelineDefinition):
        compiler.compile(pipeline=pipeline, output_path="kubeflow_pipeline.yml")


def test_kfp_import():
    """Test that the kfp import throws the correct error."""
    with mock.patch.dict(sys.modules):
        # remove kfp from the modules
        sys.modules["kfp"] = None
        with pytest.raises(ImportError):
            _ = KubeFlowCompiler()


@pytest.mark.usefixtures("_freeze_time")
def test_vertex_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to vertex."""
    example_dir, pipeline, _ = setup_pipeline
    compiler = VertexCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = VertexPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.pipeline_name == pipeline.name
        assert pipeline_configs.pipeline_description == pipeline.description
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get exepcted component configs
            component = pipeline._graph[component_name]
            component_op = component["fondant_component_op"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_vertex_configuration(tmp_path_factory):
    """Test that the kubeflow pipeline can be configured."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="NVIDIA_TESLA_K80",
        ),
    )
    pipeline.add_op(component_1)
    compiler = VertexCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = VertexPipelineConfigs.from_spec(output_path)
        component_configs = pipeline_configs.component_configs["first_component"]
        for accelerator in component_configs.accelerators:
            assert accelerator.type == "NVIDIA_TESLA_K80"
            assert accelerator.number == "1"


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_vertex_configuration(tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
    )

    pipeline.add_op(component_1)
    compiler = VertexCompiler()
    with pytest.raises(InvalidPipelineDefinition):
        compiler.compile(pipeline=pipeline, output_path="kubeflow_pipeline.yml")


def test_caching_dependency_docker(tmp_path_factory):
    """Test that the component cache key changes when a depending component cache key change for
    the docker compiler.
    """
    arg_list = ["dummy_arg_1", "dummy_arg_2"]
    second_component_cache_key_dict = {}

    for arg in arg_list:
        pipeline = Pipeline(
            pipeline_name="test_pipeline",
            pipeline_description="description of the test pipeline",
            base_path="/foo/bar",
        )
        compiler = DockerCompiler()

        component_1 = ComponentOp(
            Path(COMPONENTS_PATH / "example_1" / "first_component"),
            arguments={"storage_args": f"{arg}"},
        )
        component_2 = ComponentOp(
            Path(COMPONENTS_PATH / "example_1" / "second_component"),
            arguments={"storage_args": "a dummy string arg"},
        )

        pipeline.add_op(component_1)
        pipeline.add_op(component_2, dependencies=component_1)

        with tmp_path_factory.mktemp("temp") as fn:
            output_path = str(fn / "docker-compose.yml")
            compiler.compile(pipeline=pipeline, output_path=output_path, build_args=[])
            pipeline_configs = DockerPipelineConfigs.from_spec(output_path)
            metadata = json.loads(
                pipeline_configs.component_configs["second_component"].arguments[
                    "metadata"
                ],
            )
            cache_key = metadata["cache_key"]
            second_component_cache_key_dict[arg] = cache_key

    assert (
        second_component_cache_key_dict[arg_list[0]]
        != second_component_cache_key_dict[arg_list[1]]
    )


def test_caching_dependency_kfp(tmp_path_factory):
    """Test that the component cache key changes when a depending component cache key change for
    the kubeflow compiler.
    """
    arg_list = ["dummy_arg_1", "dummy_arg_2"]
    second_component_cache_key_dict = {}

    for arg in arg_list:
        pipeline = Pipeline(
            pipeline_name="test_pipeline",
            pipeline_description="description of the test pipeline",
            base_path="/foo/bar",
        )
        compiler = KubeFlowCompiler()

        component_1 = ComponentOp(
            Path(COMPONENTS_PATH / "example_1" / "first_component"),
            arguments={"storage_args": f"{arg}"},
        )
        component_2 = ComponentOp(
            Path(COMPONENTS_PATH / "example_1" / "second_component"),
            arguments={"storage_args": "a dummy string arg"},
        )

        pipeline.add_op(component_1)
        pipeline.add_op(component_2, dependencies=component_1)

        with tmp_path_factory.mktemp("temp") as fn:
            output_path = str(fn / "kubeflow_pipeline.yml")
            compiler.compile(pipeline=pipeline, output_path=output_path)
            pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)

            metadata = json.loads(
                pipeline_configs.component_configs["second_component"].arguments[
                    "metadata"
                ],
            )
            cache_key = metadata["cache_key"]
            second_component_cache_key_dict[arg] = cache_key
        second_component_cache_key_dict[arg] = cache_key

    assert (
        second_component_cache_key_dict[arg_list[0]]
        != second_component_cache_key_dict[arg_list[1]]
    )
