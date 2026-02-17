from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Secrets (.env) ---
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr = Field(default=None) # Optional
    gemini_api_key: SecretStr = Field(default=None)   # Optional

    # --- Tiered Model Selection ---
    # Strategic thinking (The Architect)
    planner_model: str = "gpt-4o"

    # Fast task execution (The Grunt)
    executor_model: str = "gpt-4o-mini"

    # Local exploitation (The Specialist - Ollama)
    local_model: str = "deepseek-v3:latest"
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- App Constants ---
    sandbox_container: str = "ctf-sandbox"
    max_iterations: int = 15
    knowledge_base_path: str = "./data/writeups"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()