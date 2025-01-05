# semantic_routers format Encoder
# AutoEncoder
# https://github.com/aurelio-labs/semantic-router/blob/main/semantic_router/encoders/__init__.py
# BaseEncoder
# https://github.com/aurelio-labs/semantic-router/blob/main/semantic_router/encoders/base.py
from abc import ABC
from typing import Any, Coroutine, List, Optional, Union

import backoff
import openai
from openai import OpenAIError
from openai._types import NotGiven
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, Field, validator
from pydantic.v1 import PrivateAttr
import tiktoken

class BaseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    @validator("score_threshold", pre=True, always=True)
    def set_score_threshold(cls, v):
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[Any]) -> Coroutine[Any, Any, List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    
class OpenAIEncoder(BaseEncoder):
    client: Optional[openai.Client] = None
    async_client: Optional[openai.AsyncClient] = None
    dimensions: Union[int, NotGiven] = NotGiven()
    token_limit: int = 8192  # default value, should be replaced by config
    _token_encoder: Any = PrivateAttr()
    type: str = "openai"
    max_retries: int = 3
    
    def __init__(
        self,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            score_threshold=score_threshold,
        )
        try:
            self.client = openai.Client(
                base_url=base_url, api_key=api_key
            )
            self.async_client = openai.AsyncClient(
                base_url=base_url, api_key=api_key
            )
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e
        self.name = name
        self.score_threshold = score_threshold
            
        
    @backoff.on_exception(
        backoff.expo,  # exponential backoff
        (openai.RateLimitError, openai.APIError),
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    def embed(self, docs: List[str]) -> List[float]:
        try:
            response = self.client.embeddings.create(input=docs, model=self.name)
        except OpenAIError as e:
            print(f"OpenAI Error: {e}")
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")

        if (
            not response
            or not isinstance(response, CreateEmbeddingResponse)
            or not response.data
        ):
            raise ValueError("No embeddings returned.")
        
        embeddings = [embeds_obj.embedding for embeds_obj in response.data]
        return embeddings
    
    @backoff.on_exception(
        backoff.expo,  # exponential backoff
        (openai.RateLimitError, openai.APIError),
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def aembed(self, docs: List[str]) -> List[List[float]]:
        try:
            response = await self.async_client.embeddings.create(input=docs, model=self.name)
        except OpenAIError as e:
            print(f"OpenAI Error: {e}")
            return None
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            return None
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return None
            
        if (
            not response
            or not isinstance(response, CreateEmbeddingResponse)
            or not response.data
        ):
            raise ValueError("No embeddings returned.")
        
        embeddings = [embeds_obj.embedding for embeds_obj in response.data]
        return embeddings
    
    def __call__(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using OpenAI API.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        # TODO - Implement truncate
        # if truncate:
        #     # check if any document exceeds token limit and truncate if so
        #     docs = [self._truncate(doc) for doc in docs]
        
        return self.embed(docs)
        
        
    async def acall(self, docs: List[str], truncate: bool = True) -> List[List[float]]:
        if self.async_client is None:
            raise ValueError("OpenAI async client is not initialized.")
        embeds = await self.aembed(docs)
        return embeds