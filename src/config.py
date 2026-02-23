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
    planner_model: str = "o3"
    planner_tiers: str = ""
    planner_max_tokens: int = 400

    # Fast task execution (The Grunt)
    executor_provider: str = "ollama"
    executor_model: str = "qwen2.5-coder:14b-instruct"
    executor_tiers: str = ""
    executor_max_tokens: int = 300

    # Verification / critique
    verifier_provider: str = "openai"
    verifier_model: str = "o4-mini"
    verifier_tiers: str = ""
    verifier_max_tokens: int = 400

    # --- Tier routing thresholds (comma-separated) ---
    tier_phase_cycles: str = "3,6,10"
    tier_tool_calls: str = "5,8,12"
    tier_category_pivots: str = "1,2,2"
    tier_recent_failures: str = "1,2,3"
    tier_failure_window: int = 3

    # Local exploitation (The Specialist - Ollama)
    local_model: str = "qwen2.5-coder:14b-instruct"
    ollama_base_url: str = "http://localhost:11434"

    # --- App Constants ---
    sandbox_container: str = "ctf-sandbox"
    sandbox_workdir: str = "/challenge"
    tool_timeout_seconds: int = 30
    max_wall_seconds_per_challenge: int = 600
    max_tool_calls: int = 15
    max_phase_cycles: int = 20
    max_category_pivots: int = 3
    sandbox_timeout_s: int = 30
    max_iterations: int = 20
    knowledge_base_path: str = "./data/knowledge_base"
    flag_format: str = "flag{"
    runs_dir: str = "runs"
    output_large_line_threshold: int = 300
    output_large_char_threshold: int = 20000
    output_head_lines: int = 50
    output_tail_lines: int = 50

    # --- RAG ---
    rag_mode: str = "all"
    rag_include_writeups: bool = True
    rag_enabled: bool = True
    rag_db_path: str = "./chroma_db"
    rag_embed_model: str = "BAAI/bge-base-en-v1.5"
    rag_links_db_path: str = "data/chroma"
    rag_links_collection: str = "security_kb"
    rag_writeups_collection: str = "ctf_writeups"
    rag_reference_collection: str = "ctf_reference"
    rag_doc_max_chars: int = 2000
    rag_total_max_chars: int = 7000
    debug: bool = False

    # --- Links RAG (optional CLI) ---
    links_rag_data_dir: str = "data"
    links_rag_user_agent: str = "ctf-kb-rag-ingestor/1.0 (+local)"
    links_rag_request_timeout_s: int = 30
    links_rag_http_retries: int = 2
    links_rag_crawl_delay_s: float = 0.25
    links_rag_default_max_pages: int = 80
    links_rag_default_max_depth: int = 1
    links_rag_wiki_max_pages: int = 250
    links_rag_wiki_max_depth: int = 2
    links_rag_chunk_target_tokens: int = 650
    links_rag_chunk_overlap_tokens: int = 100
    links_rag_min_chunk_tokens: int = 120
    links_rag_index_code_files: bool = False
    links_rag_max_repo_file_bytes: int = 1_500_000
    links_rag_max_text_chars_per_file: int = 120_000
    links_rag_enable_playwright: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
