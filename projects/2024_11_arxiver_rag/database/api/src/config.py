from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    app_env: Literal["dev", "prod"] = Field(default="dev")

    postgres_db: str
    postgres_url: str

@lru_cache
def get_settings():
    return Settings()