# 2501_2_vector_db
* vector db 테스트

## Experiments
### 1_langchain_search
* langchain으로 포매팅한 청크 임베딩 & 인덱싱
    * https://python.langchain.com/docs/integrations/vectorstores/qdrant/
* 실험 대상 DB 백엔드
    1. [qdrant](./docs/langchain_qdrant.md)
    2. [pgvector](./docs/langchain_pgvector.md)
    3. [sqlite-vec](./docs/langchain_sqlitevec.md)
    4. [lancedb](./docs/langchain_lancedb.md)
    5. [weaviate](./docs/langchain_weaviate.md)
    6. [milvus](./docs/langchain_milvus.md)
    7. [chromadb](./docs/langchain_chromadb.md)
    8. faiss
    9. mongodb-atlas


### 2_llamaindex_search
* llama-index based search
* 설치
    * llama-index
    * llama-index-embeddings-openai, llama-index-embeddings-langchain, llama-index-embeddings-text-embeddings-inference

#### 2-1. pgvector
* 추가 설치:
    * llama-index-vector-stores-postgres
        * langchain과 버전 충돌 주의 `langchain-postgres 0.0.12 requires pgvector<0.3.0,>=0.2.5, but you have pgvector 0.3.6 which is incompatible.`

**Notes**: [llamaindex_pgvector](./llamaindex_pgvector.md)

### 5. multi-vector retrieval
#### 5-1. llamaindex_multimodal test
* test llama-index's multimodal implementations

**Notes:** [llamaindex-multimodal](./docs/llamaindex_multimodal.md)


## Discussions
### Which VectorDB to use
Comparisons & Suggestions
* top contendors: weaviate, chromadb, pgvector, lancedb
* https://www.reddit.com/r/LocalLLaMA/comments/1e63m16/vector_database_pgvector_vs_milvus_vs_weaviate/
    * postgres fts (full-text-search)
        * https://www.postgresql.org/docs/current/textsearch.html
    * sqlite-vss working but abandoned, sqlite-vec early stages
    * suggestions:
        * pgvector + pgvectorscale
            * https://github.com/timescale/pgvectorscale
        * lancedb
        * weaviate for verbis: https://github.com/alexmavr/verbis
            * macos only
* https://www.reddit.com/r/LangChain/comments/170jigz/my_strategy_for_picking_a_vector_database_a/
    * https://benchmark.vectorview.ai/vectordbs.html

Personal Thoughts
* **pgvector (+ extensions)** seems easier for management with existing dbs
* sqlite based methods also worth looking
    * not suitable yet for large-scale
    * will eventually gain some form of ANN indexes in the near future


### Multi-vector retrieval?
* text+image embedding or colpali-style retrieval