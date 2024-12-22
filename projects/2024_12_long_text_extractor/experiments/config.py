import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'
    )

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    
@lru_cache
def get_settings():
    return Settings()


settings = get_settings()

os.environ["OPENAI_API_KEY"] = settings.openai_api_key