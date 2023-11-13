import dask
import pandas as pd
from llama_index import Document
from tqdm import tqdm

from fondant.component import PandasTransformComponent

dask.config.set({"dataframe.convert-string": False})


class CreateDocuments(PandasTransformComponent):
    
    @staticmethod
    def create_serialised_document(dataframe: pd.DataFrame):
        document = Document(
            text=dataframe[("text","data")],
            # embedding=dataframe[("text","embeddings")].tolist(),
            metadata={"source": dataframe[("text", "source")]}
            )
        return document.to_json()
    
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "documents")] = dataframe.apply(self.create_serialised_document, axis=1)
        return dataframe