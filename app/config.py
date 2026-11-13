"""
Configuration classes for the SDEWS Flask application.
Supports development, production, and testing modes.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration."""

    # ---- Security ----
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    WTF_CSRF_ENABLED = True

    # ---- Database ----
    # Saving directly in the root folder to bypass Windows permission issues
    _db_path = (BASE_DIR / "sdews.db").as_posix()
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ---- Uploads ----
    UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
    ALLOWED_EXTENSIONS = {"csv"}

    # ---- ML artifacts ----
    MODEL_DIR = BASE_DIR / "ml" / "models"

    # ---- Logging ----
    LOG_DIR = BASE_DIR / "logs"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}