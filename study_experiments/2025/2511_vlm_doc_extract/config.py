from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )
    
    DATA_DIR: str
    
    VLM_BASE_URL: str
    VLM_API_KEY: str
    VLM_MODEL: str
    
@lru_cache
def get_settings():
    return Settings()