import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

load_dotenv(Path(BASE_DIR) / ".env")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "neuroscan-dev-secret-change-me")

    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    if not SQLALCHEMY_DATABASE_URI:
        raise RuntimeError(
            "DATABASE_URL не задан. Проверьте .env в корне проекта "
            f"({BASE_DIR}/.env)"
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB на сессию
    ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
    MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.pth")
    MAX_IMAGES_PER_SESSION = 5