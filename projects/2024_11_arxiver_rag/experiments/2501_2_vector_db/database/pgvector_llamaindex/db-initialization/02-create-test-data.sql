CREATE EXTENSION vector;

-- Manually create tables that would be created by llama index
CREATE SEQUENCE IF NOT EXISTS data_test_documents_id_seq;

CREATE TABLE "public"."data_test_documents" (
    "id" int8 NOT NULL DEFAULT nextval('data_test_documents_id_seq'::regclass),
    "text" varchar NOT NULL,
    "metadata_" json,
    "node_id" varchar,
    "embedding" vector(1024),
    PRIMARY KEY ("id")
);

CREATE INDEX data_test_documents_embedding_idx ON public.data_test_documents USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64');