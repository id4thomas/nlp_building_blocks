import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from grpc import RpcError

from core.base.schema import (
    Document,
    BaseNode,
    TextNode,
    ImageNode,
    TableNode,
    doc_to_json,
    json_to_doc
)
from core.storage.vectorstore.base import BaseVectorStore
from core.storage.vectorstore.utils import document_to_metadata_dict

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.https.models import PointStruct

class QdrantLateInteractionVectorStore(BaseVectorStore):
    """
    qdrant based vectorstore for late-interaction (colbert/colpali)
    mostly follows llama-index implementation
    """
    collection_name: str
    _collection_initialized: bool = False
    _client: "QdrantClient" = None
    _aclient: "AsyncQdrantClient" = None
    
    flat_metadata: bool = True
    
    def __init__(
        self,
        collection_name: str,
        client: Optional["QdrantClient"] = None,
        aclient: Optional["AsyncQdrantClient"] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
    ):
        # Check qdrant-client import
        try:
            from qdrant_client import QdrantClient, AsyncQdrantClient
        except ImportError:
            raise ImportError("Please install qdrant-client: 'pip install qdrant-client'")
        
        if client is None and aclient is None:
            raise ValueError("Must provide either a QdrantClient instance")
        self._client = client
        self._aclient = aclient
            
        # Check if Collection Exists
        self.collection_name = collection_name
        if self._client is not None:
            self._collection_initialized = self._collection_exists(collection_name)
        else:
            #  need to do lazy init for async clients
            self._collection_initialized = False
            
        self.batch_size=batch_size
        self.parallel=parallel
        self.max_retries=max_retries
        
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self._client.collection_exists(collection_name)

    async def _acollection_exists(self, collection_name: str) -> bool:
        """Asynchronous method to check if a collection exists."""
        return await self._aclient.collection_exists(collection_name)
        
    def create(self, **kwargs):
        """Create collection"""
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        
        try:
            self._client.create_collection(
                collection_name=self.collection_name,
                **kwargs
            )
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._collection_initialized = True
    
    async def acreate(
        self
    ):
        """Asynchronously Create collection"""
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        if self._collection_initialized:
            raise ValueError(f"Collection {self.collection_name} already exists")
        
        try:
            self._aclient.create_collection(
                collection_name=self.collection_name,
            )
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            raise ValueError(f"Collection {self.collection_name} already exists")
        self._collection_initialized = True

    def _build_points(
        self,
        documents: List[BaseNode],
        embeddings: List[
            Union[List[List[float]], List[List[int]]]
        ],
        metadata_keys: Optional[List[str]] = None,
    ) -> List["PointStruct"]:
        """
        Convet documents to qdrant PointStruct instances
        Args
            documents
            embeddings: list of 2D late-interaction embedding
            metadata_keys: keys of document.metadata to store in point (default: everything)
        Returns:
            points:
        """
        from qdrant_client.http.models import PointStruct
        
        points = []
        for document, embedding in zip(documents, embeddings):
            if not (
                isinstance(embedding, list)
                and isinstance(embedding[0], list)
                and (
                    isinstance(embedding[0][0], float)
                    or isinstance(embedding[0][0], int)
                )
            ):
                raise ValueError("given embedding is not 2d")
            metadata = document_to_metadata_dict(
                document, keys=metadata_keys, flat_metadata=self.flat_metadata
            )
            point = PointStruct(
                id=document.id_,
                payload=metadata,
                vector=embedding
            )
            points.append(point)
        
        return points
    
    def add(
        self,
        documents: Union[Document, List[Document]],
        embeddings: List[
            Union[List[List[float]], List[List[int]]]
        ],
        metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Add documents to index.
        Args:
            documents: List[Document]: list of documents
            embeddings
            metadata_keys
        """
        
        if not isinstance(documents, list):
            documents = [documents]
            embeddings = [embeddings]
        if len(documents) > 0 and not self._collection_initialized:
            raise ValueError(
                f"Collection {self.collection_name} is not initialized, create it first"
            )
        if len(documents)!=len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must be same documents: {}, embeddings:{}".format(
                    len(documents), len(embeddings)
                )
            )
        points = self._build_points(
            documents=documents,
            embeddings=embeddings,
            metadata_keys=metadata_keys
        )

        self._client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
            parallel=self.parallel,
            max_retries=self.max_retries,
            wait=True,
        )
    
    async def aadd(self):
        pass
    
    def delete(self):
        pass
    
    def query(self):
        pass

    def drop(self):
        pass
    
# TODO - Implement Hybrid version
class QdrantHybridLateInteractionVectorStore(BaseVectorStore):
    """qdrant based vectorstore for late-interaction (colbert/colpali) + sparse"""
    
    def __init__(self):
        pass

    def add(self):
        pass
    
    async def aadd(self):
        pass
    
    def delete(self):
        pass
    
    def query(self):
        pass

    def drop(self):
        pass