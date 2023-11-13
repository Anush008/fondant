from typing import Optional

import dask
import dask.dataframe as dd
import pandas as pd
import weaviate
from llama_index.schema import TextNode
from llama_index.vector_stores import WeaviateVectorStore

from fondant.component import DaskWriteComponent

# from fondant.component_spec import ComponentSpec

dask.config.set({"dataframe.convert-string": False})

class WriteToVectorDB(DaskWriteComponent):
    def __init__(
            self,
            *_,
            cloud_store: bool=False,
            wcs_username: Optional[str]=None,
            wcs_password: Optional[str]=None,
            wcs_url: Optional[str]=None,
            local_url: Optional[str]=None,
            index_name: Optional[str]=None
            ):
        cloud_store = False
        if cloud_store:
            resource_owner_config = weaviate.AuthClientPassword(
                username=wcs_username,
                password=wcs_password,
            )
            self.client = weaviate.Client(url=wcs_url, auth_client_secret=resource_owner_config)
        else:
            self.client = weaviate.Client(url=local_url)
        self.vector_store = WeaviateVectorStore(weaviate_client=self.client, index_name=index_name)

    def deserialise_chunks(self, nodes: pd.Series):
        return TextNode.from_json(nodes)
    
    def load_partitions(self, df, vector_store):
        nodes = df[("text_node")].apply(self.deserialise_chunks)
        vector_store.add(nodes)

    def write(self, dataframe: dd.DataFrame):
        vector_store = self.vector_store

        dataframe.map_partitions(
            self.load_partitions,
            vector_store=vector_store,
            meta=("nodes", "object"),
        ).compute()
