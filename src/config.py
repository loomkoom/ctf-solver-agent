from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Secrets (.env) ---
    openai_api_key: SecretStr | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    gemini_api_key: SecretStr | None = Field(default=None)

    # --- CTFd ---
    ctfd_url: str = ""
    ctfd_token: SecretStr | None = Field(default=None)
    ctfd_username: str = ""
    ctfd_password: SecretStr | None = Field(default=None)
    ctfd_verify_tls: bool = True
    ctfd_timeout_s: int = 15

    # --- Tiered Model Selection ---
    # Strategic thinking (The Architect)
    planner_provider: str = "openai"
    planner_model: str = "gpt-4o"

    # Fast task execution (The Grunt)
    executor_provider: str = "openai"
    executor_model: str = "gpt-4o-mini"

    # Local exploitation (The Specialist - Ollama)
    local_model: str = "deepseek-v3:latest"
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- App Constants ---
    sandbox_container: str = "ctf-sandbox"
    sandbox_workdir: str = "/challenge"
    sandbox_timeout_s: int = 30
    max_iterations: int = 15
    knowledge_base_path: str = "./data/knowledge_base"
    flag_format: str = "flag{"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
