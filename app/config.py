"""Settings from environment / .env (pydantic-settings)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # WHY sqlite default: zero-setup local run; production/CI use Postgres via DATABASE_URL.
    database_url: str = "sqlite:///./sentinel.db"
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    embedding_provider: str = "hash"
    tool_transport: str = "mcp"
    max_steps: int = 8
    # Rough USD per 1M tokens, only for the audit-log cost estimate.
    price_in_per_m: float = 3.0
    price_out_per_m: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
