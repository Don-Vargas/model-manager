# app/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "model-manager"

    database_url: str = "sqlite:////data/model-manager.db"

    hf_token: str | None = None

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()