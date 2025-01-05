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
    * https://python.langchain.com/docs/integrations/vectorstores/sqlitevec/
* 추가 설치
    * sqlite-vec
* issues
    * update & delete on vector store not supported yet via langchain
    * query->retriever not supported yet


### 2_llamaindex_search
* llama-index based search

#### 2-1. pgvector
* db:
* 추가 설치:
    * llama-index-vector-stores-postgres, llama-index


## Database
* database tests