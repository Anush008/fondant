import itertools

import dask
import pandas as pd
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from tqdm import tqdm

from fondant.component import PandasTransformComponent

dask.config.set({"dataframe.convert-string": False})


# class CreateNodes(PandasTransformComponent):
#     def __init__(self, *_, chunk_size: int):
#         self.parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)

#     def deserialise_documents(self, documents: pd.Series):
#         return Document.from_json(documents)
    
#     def chunk_documents(self, dataframe: pd.DataFrame):
#         return self.parser.get_nodes_from_documents(documents=dataframe[("text", "documents")].apply(self.deserialise_documents))
    
#     def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
#         nodes = self.chunk_documents(dataframe)
#         nodes_df = pd.DataFrame(
#             {
#                 "nodes": [node.to_json() for node in nodes]
#             }
#         )
#         nodes_df.index.name = "id"
#         nodes_df.index = nodes_df.index.map(str)
#         # Set multi-index column for the expected subset and field
#         nodes_df.columns = pd.MultiIndex.from_product(
#             [["chunks"], nodes_df.columns],
#         )
#         return nodes_df
    
class CreateNodes(PandasTransformComponent):
    def __init__(self, *_, chunk_size: int, chunk_overlap: int):
        self.parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def deserialise_documents(self, documents: str):
        return Document.from_json(documents)

    def chunk_documents(self, row):
        # id, text, metadata, nodes
        document = [self.deserialise_documents(row[("text", "documents")])]
        doc_id = document[0].id_
        nodes = self.parser.get_nodes_from_documents(documents=document)
        node_list = []
        for node in nodes:
            node_id = node.id_
            chunk = node.text
            source = node.metadata['source']
            node = node.to_json()
            node_list.append((doc_id, node_id, chunk, source, node))
        return node_list

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Return a pandas series of chunked documents
        results = dataframe.apply(
            self.chunk_documents,
            axis=1,
        ).to_list()

        # Flatten result
        results = list(itertools.chain.from_iterable(results))

        # Turn into dataframes
        results_df = pd.DataFrame(
            results,
            columns=["original+document+id", "node_id", "chunk", "source", "node"],
        )
        results_df = results_df.set_index("node_id")

        # Set multi-index column for the expected subset and field
        results_df.columns = pd.MultiIndex.from_product(
            [["text"], results_df.columns],
        )

        return results_df