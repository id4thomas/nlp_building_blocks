# 2501_2_vector_db
* vector db 테스트

## Experiments
### 1_langchain_search
* langchain으로 포매팅한 청크 임베딩 & 인덱싱
    * https://python.langchain.com/docs/integrations/vectorstores/qdrant/
* 설치 패키지:
    * langchain-qdrant, langchain-openai
    * fastembed
#### 1-1. qdrant
* db: qdrant in-memory

#### 1-2. pgvector
* db: pgvector
    * https://python.langchain.com/docs/integrations/vectorstores/pgvector/
    * https://github.com/pgvector/pgvector
* 추가 설치
    * psycopg[binary,pool], langchain_postgres
    * brew install libpq

* langchain combines all 'collections' rows into 2 tables
    * https://github.com/langchain-ai/langchain/issues/6045
    * `langchain_pg_collection`, `langchain_pg_embedding`

**db schema used by langchain**

`langchain_pg_collection`
```
CREATE TABLE "public"."langchain_pg_collection" (
    "uuid" uuid NOT NULL,
    "name" varchar NOT NULL,
    "cmetadata" json,
    PRIMARY KEY ("uuid")
);
CREATE UNIQUE INDEX langchain_pg_collection_name_key ON public.langchain_pg_collection USING btree (name);
```

`langchain_pg_embedding`
```
CREATE TABLE "public"."langchain_pg_embedding" (
    "id" varchar NOT NULL,
    "collection_id" uuid,
    "embedding" vector,
    "document" varchar,
    "cmetadata" jsonb,
    CONSTRAINT "langchain_pg_embedding_collection_id_fkey" FOREIGN KEY ("collection_id") REFERENCES "public"."langchain_pg_collection"("uuid") ON DELETE CASCADE,
    PRIMARY KEY ("id")
);
CREATE INDEX ix_cmetadata_gin ON public.langchain_pg_embedding USING gin (cmetadata jsonb_path_ops);
CREATE UNIQUE INDEX ix_langchain_pg_embedding_id ON public.langchain_pg_embedding USING btree (id);
```

#### 1-3. sqlite-vec
* db: sqlite + sqlite-vec
    * https://alexgarcia.xyz/sqlite-vec/
        * !! Only brute-force search for now
        * 
    * https://python.langchain.com/docs/integrations/vectorstores/sqlitevec/
* 추가 설치
    * sqlite-vec
* issues
    * update & delete on vector store not supported yet via langchain
    * query->retriever not supported yet

#### 1-4. lancedb
* db: lancedb
    * https://python.langchain.com/docs/integrations/vectorstores/lancedb/
    * https://lancedb.github.io/lancedb/basic/#preview-releases
* 추가 설치:
    * pip install tantivy lancedb

**Notes**
persistent storage folder:
```
.
└── langchain_test.lance
    ├── _transactions
    ├── _versions
    └── data
```

#### 1-5. weaviate
* db: weaviate (running local container)
    * https://python.langchain.com/docs/integrations/vectorstores/weaviate/
    * https://weaviate.io/developers/weaviate/installation/docker-compose
* 추가 설치:
    * pip install langchain-weaviate
        * `Detected incompatible Protobuf Gencode/Runtime versions when loading grpc_health/v1/health.proto: gencode 5.29.0 runtime 5.27.1. Runtime version cannot be older than the linked gencode version. See Protobuf version guarantees at` -> update protobuf 

**Notes**
* no metadata filter support

persistent storage folder:
```
(base) ➜  local_storage git:(main) ✗ tree -L 2
.
├── classifications.db
├── langchain_04480efc670c4256ab58d017cb184770
│   └── n3Et04Myjnkd
├── migration1.19.filter2search.skip.flag
├── migration1.19.filter2search.state
├── migration1.22.fs.hierarchy
├── modules.db
├── raft
│   ├── raft.db
│   └── snapshots
└── schema.db
```

### 2_llamaindex_search
* llama-index based search
* 설치
    * llama-index
    * llama-index-embeddings-openai, llama-index-embeddings-langchain, llama-index-embeddings-text-embeddings-inference

#### 2-1. pgvector
* db:
* 추가 설치:
    * llama-index-vector-stores-postgres
        * langchain과 버전 충돌 주의 `langchain-postgres 0.0.12 requires pgvector<0.3.0,>=0.2.5, but you have pgvector 0.3.6 which is incompatible.`

## Database
* database tests

### discussions
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