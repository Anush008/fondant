import dask
import dask.dataframe as dd
import pandas as pd
import weaviate
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import WeaviateVectorStore

from fondant.component import DaskWriteComponent

# from fondant.component_spec import ComponentSpec

dask.config.set({"dataframe.convert-string": False})


class WriteToVectorDB(DaskWriteComponent):
    def __init__(self, *_, wcs_username: str, wcs_password: str, wcs_url: str):
        resource_owner_config = weaviate.AuthClientPassword(
            username=wcs_username,
            password=wcs_password,
        )
        client = weaviate.Client(url=wcs_url, auth_client_secret=resource_owner_config)
        self.vector_store = WeaviateVectorStore(client)

    def build_documents(self, dataframe: pd.DataFrame) -> Document:
        return [
            Document(
                text=el["text_data"],
                metadata={"source": el["text_source"]},
                embedding=el[
                    "text_embeddings"
                ].tolist(),  # must be a list for Llama-index Document
            )
            for i, el in dataframe.iterrows()
        ]

    def parse_nodes(self, documents: list):
        parser = (
            SimpleNodeParser.from_defaults()
        )  # default chunk size = 1024, chunk overlap = 20
        return parser.get_nodes_from_documents(documents=documents)

    def load_partitions(self, df, vector_store):
        documents = self.build_documents(df)
        nodes = self.parse_nodes(documents=documents)
        vector_store.add(nodes)

    def write(self, dataframe: dd.DataFrame):
        vector_store = self.vector_store

        # iterate through partitions
        # for part in dataframe.partitions:
        #     documents = self.build_documents(dataframe=part)
        #     nodes = self.parse_nodes(documents=documents)

        #     vector_store.add(nodes)

        # or map partition
        dataframe.map_partitions(
            self.load_partitions,
            vector_store=vector_store,
            meta=(("data", "source", "embeddings"), "object"),
        ).compute()
