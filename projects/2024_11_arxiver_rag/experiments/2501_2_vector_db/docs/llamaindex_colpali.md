# llamaindex + colpali (late-interaction)
## Envs
* llama-index v0.12.15

## Resources
* Reranking using ColPali, Cohere Reranker and Multi-Modal Embeddings
    * https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/ColPaliRerank/#reranking-using-colpali-cohere-reranker-and-multi-modal-embeddings
    * **Doesn't use colpali embedding for first-stage retrieval**
    * use cohere multimodal embedding (embed-english-v3.0)
    * retrieves text / image nodes separately with MultiModalVectorIndexRetriever
    * ranks text/image nodes separately
        * text with cohere reranker
        * image with colpali reranker