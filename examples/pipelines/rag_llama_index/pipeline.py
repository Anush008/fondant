import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
sys.path.append("../")
from fondant.pipeline import ComponentOp, Pipeline


def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)


BASE_PATH = "./text"
BASE_PATH = create_directory_if_not_exists(BASE_PATH)

# define pipeline
pipeline = Pipeline(pipeline_name="rag_llama_index", base_path=BASE_PATH)

# load from hub component
load_component_column_mapping = {
    "chunk-id": "text_chunk-id",
    "chunk": "text_data",
    "source": "text_source",
}

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "jamescalam/llama-2-arxiv-papers-chunked@~parquet",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 200,
    },
)

# get documents
get_documents = ComponentOp(
    component_dir="components/create_documents"
)

# get nodes
get_nodes = ComponentOp(
    component_dir="components/create_nodes",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    cache=False
)

# get embeddings
get_embeddings = ComponentOp(
    component_dir="components/get_embeddings",
    arguments={"hf_embed_model": "BAAI/bge-small-en"},
    # number_of_gpus=1,
)

# write to weaviate
write_index = ComponentOp(
    component_dir="components/write_to_vector_db",
    arguments={
        "cloud_store": False,
        "wcs_username": None,
        "wcs_password": None,
        "wcs_url": None,
        "local_url": "http://host.docker.internal:8080",
        "index_name": "Llama_Paper"
    },
)
# add components
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(get_documents, dependencies=[load_from_hf_hub])
pipeline.add_op(get_nodes, dependencies=[get_documents])
pipeline.add_op(get_embeddings, dependencies=[get_nodes])
pipeline.add_op(write_index, dependencies=[get_embeddings])
