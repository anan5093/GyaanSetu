import os


def get_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ["true", "1", "yes"]


def get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


def get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except ValueError:
        return default


class Settings:
    # ===============================
    # 📘 Dataset Config
    # ===============================
    CLASS_ID: int = get_int("CLASS_ID", 10)
    SUBJECT: str = os.getenv("SUBJECT", "science")

    # ===============================
    # 🔎 Retrieval Config
    # ===============================
    TOP_K: int = get_int("TOP_K", 3)
    SCORE_THRESHOLD: float = get_float("SCORE_THRESHOLD", 0.3)

    # ===============================
    # 🤖 LLM Config
    # ===============================
    USE_MOCK_LLM: bool = get_bool("USE_MOCK_LLM", "true")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "nvidia/nemotron-3-super:free"
    )
    MAX_TOKENS: int = get_int("MAX_TOKENS", 300)

    # ===============================
    # ⚡ API Config
    # ===============================
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = get_int("API_PORT", 8000)

    # ===============================
    # 🌐 Frontend Connection
    # ===============================
    FRONTEND_URL: str = os.getenv(
        "FRONTEND_URL",
        "http://localhost:5173"
    )

    # ===============================
    # 🧪 Debug / Logging
    # ===============================
    ENABLE_LOGS: bool = get_bool("ENABLE_LOGS", "true")


# ✅ Singleton instance
settings = Settings()
