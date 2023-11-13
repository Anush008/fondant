import dask
import pandas as pd
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode
from tqdm import tqdm

from fondant.component import PandasTransformComponent

dask.config.set({"dataframe.convert-string": False})


class GenerateEmbeddings(PandasTransformComponent):
    def __init__(self, *_, hf_embed_model: str) -> None:
        """
        Args:
            argumentX: An argument passed to the component
        """
        # Initialize your component here based on the arguments
        self.model = HuggingFaceEmbedding(hf_embed_model)

    def deserialise_nodes(self, node: str):
        return TextNode.from_json(node)
    
    def get_embeddings(self, node):
        node = self.deserialise_nodes(node[("text", "node")])
        node_embedding = self.model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
        embedding = node.embedding
        return node.to_json(), embedding

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[[("text", "node"), ("text", "embeddings")]] = dataframe.apply(self.get_embeddings, axis=1, result_type='expand')
        return dataframe