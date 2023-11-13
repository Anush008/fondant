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
pipeline = Pipeline(pipeline_name="embeddings-llama2-paper", base_path=BASE_PATH)

# load for Google docs component
data_loader = ComponentOp(
    component_dir="components/data_loader",
    arguments={
        "folder_url": [
            "https://drive.google.com/drive/u/0/folders/13aWtM55FUZksGTOt1FtkrFoy3QROxR3q"
        ],
        "docs_url": [],
        "is_folder": True,
    },
)

# add components
pipeline.add_op(data_loader)
