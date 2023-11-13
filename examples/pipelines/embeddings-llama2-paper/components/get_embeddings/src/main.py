import dask
import pandas as pd
from llama_index.embeddings import HuggingFaceEmbedding
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

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "embeddings")] = [
            self.model.get_text_embedding(text)
            for i, text in tqdm(enumerate(dataframe[("text", "data")]))
        ]
        return dataframe
