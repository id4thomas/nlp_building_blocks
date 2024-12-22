from typing import List, Union, TYPE_CHECKING

import backoff
import openai

from openai import AsyncOpenAI
from config import settings


if TYPE_CHECKING:
    from pydantic import BaseModel


class OpenAIClient:
    def __init__(self, organization: str = None, project: str = None, base_url: str = None):
        self.organization = organization
        self.project = project

        client_params = {
            "api_key": settings.openai_api_key,
            "organization": self.organization,
            "project": self.project,
        }

        if base_url:
            client_params["base_url"] = base_url

        self.client = AsyncOpenAI(**client_params)

    @backoff.on_exception(
        backoff.expo,  # exponential backoff
        (openai.RateLimitError, openai.APIError, openai.LengthFinishReasonError),
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def chat_completion_with_json_schema(
        self,
        response_format: "BaseModel",
        messages: list,
        model: str = "gpt-4o-mini",
        decode_params: dict = None,
    ):
        decode_params = decode_params or {}
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            **decode_params,
        )

        return response.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,  # exponential backoff
        (openai.RateLimitError, openai.APIError),
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def chat_completion(self, messages: list, model: str = "gpt-4o-mini", decode_params: dict = None):
        decode_params = decode_params or {}

        response = await self.client.chat.completions.create(model=model, messages=messages, **decode_params)

        return response.choices[0].message.content

    async def embedding(
        self, texts: Union[str, List[str]], model: str = "text-embedding-3-small", dimensions: int = None
    ):
        """Return Embedding object"""

        if dimensions is None:
            response = await self.client.embeddings.create(model=model, input=texts)
        else:
            response = await self.client.embeddings.create(model=model, input=texts, dimensions=dimensions)

        return response.data