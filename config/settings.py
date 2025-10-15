"""ConfigurationManagementModule"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """ApplicationConfiguration"""
    
    serper_api_key: Optional[str] = Field(default=None, env="SERPER_API_KEY")
    jina_api_key: Optional[str] = Field(default=None, env="JINA_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    
    searxng_instance_url: Optional[str] = Field(default=None, env="SEARXNG_INSTANCE_URL")
    searxng_api_key: Optional[str] = Field(default=None, env="SEARXNG_API_KEY")
    
    litellm_model_id: str = Field(
        default="openrouter/google/gemini-2.0-flash-001",
        env="LITELLM_MODEL_ID"
    )
    litellm_orchestrator_model_id: str = Field(
        default="openrouter/google/gemini-2.0-flash-001",
        env="LITELLM_ORCHESTRATOR_MODEL_ID"
    )
    litellm_eval_model_id: str = Field(
        default="gpt-4o-mini",
        env="LITELLM_EVAL_MODEL_ID"
    )
    
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-001",
        env="OPENROUTER_MODEL"
    )
    your_site_url: Optional[str] = Field(default=None, env="YOUR_SITE_URL")
    your_site_name: Optional[str] = Field(
        default="Prediction AI Agent",
        env="YOUR_SITE_NAME"
    )
    
    search_provider: str = Field(default="serper", env="SEARCH_PROVIDER")
    reranker: str = Field(default="jina", env="RERANKER")
    
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    database_url: str = Field(
        default="sqlite:///./prediction_agent.db",
        env="DATABASE_URL"
    )
    
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    prediction_confidence_threshold: float = Field(
        default=0.6,
        env="PREDICTION_CONFIDENCE_THRESHOLD"
    )
    kelly_fraction: float = Field(default=0.25, env="KELLY_FRACTION")
    max_bet_percentage: float = Field(default=0.05, env="MAX_BET_PERCENTAGE")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/agent.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()


def get_settings() -> Settings:
    """GetConfigurationInstance"""
    return settings

