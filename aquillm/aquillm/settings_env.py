"""Environment parsers shared by Django configuration."""
import os


def env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_csv(key: str) -> list[str]:
    value = os.environ.get(key, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def env_int(key: str, default: int) -> int:
    try:
        value = int((os.environ.get(key) or str(default)).strip())
    except Exception:
        return default
    return value if value >= 0 else default


def env_float(key: str, default: float) -> float:
    try:
        value = float((os.environ.get(key) or str(default)).strip())
    except Exception:
        return default
    return value if value >= 0 else default


def env_kg_int(key: str, default: int) -> int | str:
    """Preserve invalid KG values so the bounded retrieval seam can reject them."""

    raw = os.environ.get(key)
    if raw is None:
        return default
    value = raw.strip()
    try:
        return int(value, 10)
    except ValueError:
        return value


def env_kg_float(key: str, default: float) -> float | str:
    """Parse numeric KG values without hiding malformed or non-finite input."""

    raw = os.environ.get(key)
    if raw is None:
        return default
    value = raw.strip()
    try:
        return float(value)
    except ValueError:
        return value
