# 2412_1_section_chunking
* semantic-chunking 기반의 섹션 청킹기 개발

## Experiments
### 1_test_chunking_formatting
* sample 100개 정도 처리 후 청크 구성 테스트
* semantic_chunkers 패키지의 `StatisticalChunker`로 청킹
    * https://www.aurelio.ai/learn/semantic-chunkers-intro
    * 유사도 비교 모델로 baai/bge-m3 사용 (text-embedding-inference로 띄워둠)
* 청킹 후 짧은 청크(<128 토큰)는 앞 텍스트로 머징
* 섹션 제목 활용해서 포매팅

포매팅 템플릿
* p_chunk_template: Subsection이 'p' header일 경우
* h3_chunk_template: Subsection이 'h3' header일 경우
```
p_chunk_template = '''Section Title: "{section_title}"
Text:
{text}'''

h3_chunk_template = '''Section Title: "{section_title}"
Subsection Title: "{subsection_title}"
Text:
{text}'''
```

### 2_langchain_search
* langchain으로 포매팅한 청크 임베딩 & 인덱싱
    * https://python.langchain.com/docs/integrations/vectorstores/qdrant/
* 설치 패키지:
    * langchain-qdrant, langchain-openai
    * fastembed
### 2-1. qdrant
* db: qdrant in-memory
### 2-2. pgvector
* db: pgvector
    * https://python.langchain.com/docs/integrations/vectorstores/pgvector/
    * https://github.com/pgvector/pgvector
* 추가 설치
    * psycopg[binary,pool], langchain_postgres
    * brew install libpq

* langchain combines all 'collections' rows into 2 tables
    * https://github.com/langchain-ai/langchain/issues/6045
    * `langchain_pg_collection`, `langchain_pg_embedding`

#### db schema used by langchain

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

### 3_direct_pgvector_test
* test interacting with pgvector directly