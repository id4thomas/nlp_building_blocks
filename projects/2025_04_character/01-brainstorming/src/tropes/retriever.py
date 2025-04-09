from typing import List

from llama_index.core import (
	Document,
	SimpleDirectoryReader, VectorStoreIndex,
	StorageContext,
	Settings,
	QueryBundle
)
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)
from qdrant_client import QdrantClient

from src.tropes.models import Trope

class TropeRetriever:
    def __init__(self, client: QdrantClient):
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name="tropes"
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
    def get_retriever(self, k: int = 5):
        retriever_args = {"similarity_top_k": k}
        retriever = self.index.as_retriever(**retriever_args)
        return retriever

    def retrieve(self, query, k: int = 5) -> List[Trope]:
        retriever = self.get_retriever(k=k)
        results = retriever.retrieve(query)
        
        tropes = []
        for result in results:
            trope_id = result.node.metadata["TropeID"]
            name, description = result.node.text.split("\n", 1)
            
            trope = Trope(
                trope_id=trope_id,
                name=name[7:],
                explanation=description.strip()
            )
            tropes.append(trope)
        return tropes
            
    