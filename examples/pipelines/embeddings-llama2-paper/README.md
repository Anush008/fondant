# Generate embeddings pipeline

Generates embedding based on a paper over Llama2 and load them in a Weaviate vectorDB using LlamaIndex. The paper has already been chunked and prepared by jamescalam. It contains 4838 rows.

HF repository: jamescalam/llama-2-arxiv-papers-chunked

To run the pipeline, you must first create a [WCS](https://weaviate.io/developers/wcs/quickstart). Then, in the write_index ComponentOp, specify the WCS username, password, and cluster url.

## Component 1

Load data from HF using a reusable load component. 

## Component 2 

Import embed model and create an embedding column using a transform component.

## Component 3 

Create LlamaIndex instances ([Documents and Nodes](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/documents_and_nodes/root.html)) and add them to a [vector store](https://docs.llamaindex.ai/en/stable/core_modules/data_modules/index/index_guide.html#vector-store-index) using a writing component.