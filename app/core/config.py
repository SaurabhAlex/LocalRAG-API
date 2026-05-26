from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Storage
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    faiss_dir: str = "data/faiss"
    memory_dir: str = "data/memory"


    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Retrieval
    top_k: int = 5

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_temperature: float = 0.2
    ollama_num_ctx: int = 2048

    # Memory
    memory_max_turns: int = 8


settings = Settings()

