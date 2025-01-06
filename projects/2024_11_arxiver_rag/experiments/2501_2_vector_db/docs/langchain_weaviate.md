# langchain + weaviate
* no metadata filter support

## Envs
* db: weaviate (running local container)
    * https://python.langchain.com/docs/integrations/vectorstores/weaviate/
    * https://weaviate.io/developers/weaviate/installation/docker-compose
* 추가 설치:
    * pip install langchain-weaviate
        * `Detected incompatible Protobuf Gencode/Runtime versions when loading grpc_health/v1/health.proto: gencode 5.29.0 runtime 5.27.1. Runtime version cannot be older than the linked gencode version. See Protobuf version guarantees at` -> update protobuf 

## Storage Structure
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