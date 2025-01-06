# langchain + sqlite-vec

## Envs
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

## Schemas
* when initializing `SQLiteVec` vector store with table_name `test_table`
* 6 tables unique for `table_name` are created
    * `{table_name}`
    * `{table_name}_vec`
    * `{table_name}_vec_info`
    * `{table_name}_vec_chunks`
    * `{table_name}_vec_rowids`
    * `{table_name}_vec_vector_chunks00`
* `sqlite_sequence`: keep track of the last used AUTOINCREMENT values for each table
```
-- Schema for table: sqlite_sequence
CREATE TABLE sqlite_sequence(name,seq);

-- Schema for table: test_table
CREATE TABLE test_table(
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    metadata BLOB,
    text_embedding BLOB
);

-- Schema for table: test_table_vec
CREATE VIRTUAL TABLE test_table_vec USING vec0(
    rowid INTEGER PRIMARY KEY,
    text_embedding float[1024]
);

-- Schema for table: test_table_vec_info
CREATE TABLE "test_table_vec_info" (key text primary key, value any);

-- Schema for table: test_table_vec_chunks
CREATE TABLE "test_table_vec_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);

-- Schema for table: test_table_vec_rowids
CREATE TABLE "test_table_vec_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id,chunk_id INTEGER,chunk_offset INTEGER);

-- Schema for table: test_table_vec_vector_chunks00
CREATE TABLE "test_table_vec_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);
```


### {table_name}_vec_info
```
{
    "CREATE_VERSION": "v0.1.6",
    "CREATE_VERSION_MAJOR": 0,
    "CREATE_VERSION_MINOR": 1,
    "CREATE_VERSION_PATCH": 6
}
```