from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    '''OpenAI compatible LLM Setting'''
    base_url: str = Field(..., description="OpenAI compatible base url")
    api_key: str = Field(..., description="api key")
    model: str = Field(..., description="LLM Model")

class LocationMCPSettings(BaseSettings):
    url: str = Field(..., description="Location tool mcp server url")
    
class WeatherMCPSettings(BaseSettings):
    url: str = Field(..., description="Location tool mcp server url")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )
    
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    location_mcp: LocationMCPSettings = Field(default_factory=LocationMCPSettings)
    weather_mcp: WeatherMCPSettings = Field(default_factory=WeatherMCPSettings)
    
    
@lru_cache
def get_settings():
    return Settings()
