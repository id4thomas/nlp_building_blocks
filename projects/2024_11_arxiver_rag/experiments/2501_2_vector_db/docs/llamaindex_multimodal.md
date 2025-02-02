# llamaindex-multimodal
## Envs
* llama-index v0.12.15

## Analysis
### MultiModalVectorIndex
index both text and images into underlying storage systems

During retrieval-time:
* Retrieve text via vector search on the text embeddings
* Retrieve images via vector search on the image embeddings
* Both text and images are returned as Nodes in the result list -> synthesize

#### Implementation

MultiModalVectorStoreIndex usage:
* use separate stores for text/image
* https://github.com/run-llama/llama_index/blob/89396ae11da5dba7a99c95024137342d3613f06f/llama-index-core/llama_index/core/indices/multi_modal/base.py#L42
    * `image_vector_store going to be deprecated. image_store can be passed from storage_context`
```
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

MultiModalVectorStoreIndex.__add_nodes_to_index:
* https://github.com/run-llama/llama_index/blob/89396ae11da5dba7a99c95024137342d3613f06f/llama-index-core/llama_index/core/indices/multi_modal/base.py#L350C9-L350C28
* separately embed text / images
```
def _get_node_with_embedding(...):
    if is_image:
        assert all(isinstance(node, ImageNode) for node in nodes)
        id_to_embed_map = embed_image_nodes(
            nodes,  # type: ignore
            embed_model=self._image_embed_model,
            show_progress=show_progress,
        )
        ...

def _add_nodes_to_index(...):
    for node in nodes:
        if isinstance(node, ImageNode):
            image_nodes.append(node)
        if isinstance(node, TextNode) and node.text:
            text_nodes.append(node)
```


### MultiModalRetriever,

#### Implementation
* MultiModalRetriever (base class)
    * https://github.com/run-llama/llama_index/blob/89396ae11da5dba7a99c95024137342d3613f06f/llama-index-core/llama_index/core/base/base_multi_modal_retriever.py#L11

* MultiModalVectorIndexRetriever
    * https://github.com/run-llama/llama_index/blob/89396ae11da5dba7a99c95024137342d3613f06f/llama-index-core/llama_index/core/indices/multi_modal/retriever.py#L34
    * retrieve method returns text & image retrieval results separately
        * needs separate reranking

```
# retrieve method
def _retrieve(
    self,
    query_bundle: QueryBundle,
) -> List[NodeWithScore]:
    res = []
    # If text vector store is not empty, retrieve text nodes
    # If text vector store is empty, please create index without text vector store
    if self._vector_store is not None:
        res.extend(self._text_retrieve(query_bundle))

    # If image vector store is not empty, retrieve text nodes
    # If image vector store is empty, please create index without image vector store
    if self._image_vector_store is not None:
        res.extend(self._text_to_image_retrieve(query_bundle))
    return res
```


## Resources
* Multi-Modal RAG (2023.11)
    * https://www.llamaindex.ai/blog/multi-modal-rag-621de7525fea
    * Multi-Modal Embeddings:
        * `MultiModalEmbedding`: embedder base class
        * `MultiModalVectorIndex`: index both text and images into underlying storage systems
* Documentation (beta as of v0.12.15)
    * https://docs.llamaindex.ai/en/stable/module_guides/models/multi_modal/
    * uses chromadb for examples