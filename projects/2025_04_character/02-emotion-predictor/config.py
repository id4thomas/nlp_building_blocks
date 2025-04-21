from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )
    data_dir: str
    
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_model_dir: str
    
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    
    openai_api_key:str

class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    vllm_base_url: str
    vllm_api_key: str
    
settings = Settings()
experiment_settings = ExperimentSettings()