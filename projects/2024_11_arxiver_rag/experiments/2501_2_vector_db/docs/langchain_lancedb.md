# langchain + lancedb

## Envs
* db: lancedb
    * https://python.langchain.com/docs/integrations/vectorstores/lancedb/
    * https://lancedb.github.io/lancedb/basic/#preview-releases
* 추가 설치:
    * pip install tantivy lancedb

## Storage Structure
persistent storage folder:
```
.
└── langchain_test.lance
    ├── _transactions
    ├── _versions
    └── data
```