## Read Configs
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    
    embedding_base_url: str
    embedding_model: str

settings = Settings()