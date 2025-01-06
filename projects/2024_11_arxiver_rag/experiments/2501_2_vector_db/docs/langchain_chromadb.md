# langchain + chromadb
* memory usage
    * https://github.com/chroma-core/chroma/issues/1323
        * Vectors - **this gets loaded in memory** and the formula for calculating memory requirements is: 4 bytes * num vectors * vector dimensionality
        * Metadata - these are your documents and metadata, stored in SQLite DB and dynamically loaded in memory as needed (depending on your queries)
    * https://cookbook.chromadb.dev/strategies/memory-management/#lru-cache-strategy
        * LRU cache strategy which unloads segments (collections) that are not used

## Envs


## Storage Structure
persistent storage folder:
* `{uuid}` folders each correspond to a collection
```
.
├── afe9ba96-b4d8-4456-9e62-8f433b357da6
│   ├── data_level0.bin
│   ├── header.bin
│   ├── length.bin
│   └── link_lists.bin
├── chroma.sqlite3
└── d0e11c2f-d5d6-46ee-a038-e6000732db13
    ├── data_level0.bin
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

### Schema (chroma.sqlite3)
```
-- Schema for table: migrations
CREATE TABLE migrations (
                        dir TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        filename TEXT NOT NULL,
                        sql TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        PRIMARY KEY (dir, version)
                    );

-- Schema for table: embeddings_queue
CREATE TABLE embeddings_queue (
    seq_id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    operation INTEGER NOT NULL,
    topic TEXT NOT NULL,
    id TEXT NOT NULL,
    vector BLOB,
    encoding TEXT,
    metadata TEXT
);

-- Schema for table: embeddings_queue_config
CREATE TABLE embeddings_queue_config (
    id INTEGER PRIMARY KEY,
    config_json_str TEXT
);

-- Schema for table: collection_metadata
CREATE TABLE collection_metadata (
    collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (collection_id, key)
);

-- Schema for table: segment_metadata
CREATE TABLE segment_metadata (
    segment_id TEXT  REFERENCES segments(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (segment_id, key)
);

-- Schema for table: tenants
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    UNIQUE (id)
);

-- Schema for table: databases
CREATE TABLE databases (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per tenant
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    UNIQUE (tenant_id, name) -- Ensure that a tenant has only one database with a given name
);

-- Schema for table: collections
CREATE TABLE "collections" (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per database
    dimension INTEGER,
    database_id TEXT NOT NULL REFERENCES databases(id) ON DELETE CASCADE, config_json_str TEXT,
    UNIQUE (name, database_id)
);

-- Schema for table: maintenance_log
CREATE TABLE maintenance_log (
  id INT PRIMARY KEY,
  timestamp INT NOT NULL,
  operation TEXT NOT NULL
);

-- Schema for table: segments
CREATE TABLE "segments" (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    collection TEXT REFERENCES collection(id) NOT NULL
);

-- Schema for table: embeddings
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    segment_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    seq_id BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, embedding_id)
);

-- Schema for table: embedding_metadata
CREATE TABLE embedding_metadata (
    id INTEGER REFERENCES embeddings(id),
    key TEXT NOT NULL,
    string_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (id, key)
);

-- Schema for table: max_seq_id
CREATE TABLE max_seq_id (
    segment_id TEXT PRIMARY KEY,
    seq_id BLOB NOT NULL
);

-- Schema for table: embedding_fulltext_search
CREATE VIRTUAL TABLE embedding_fulltext_search USING fts5(string_value, tokenize='trigram');

-- Schema for table: embedding_fulltext_search_data
CREATE TABLE 'embedding_fulltext_search_data'(id INTEGER PRIMARY KEY, block BLOB);

-- Schema for table: embedding_fulltext_search_idx
CREATE TABLE 'embedding_fulltext_search_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;

-- Schema for table: embedding_fulltext_search_content
CREATE TABLE 'embedding_fulltext_search_content'(id INTEGER PRIMARY KEY, c0);

-- Schema for table: embedding_fulltext_search_docsize
CREATE TABLE 'embedding_fulltext_search_docsize'(id INTEGER PRIMARY KEY, sz BLOB);

-- Schema for table: embedding_fulltext_search_config
CREATE TABLE 'embedding_fulltext_search_config'(k PRIMARY KEY, v) WITHOUT ROWID;
```