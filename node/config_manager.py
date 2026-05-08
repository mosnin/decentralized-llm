import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


@dataclass
class ConfigSchema:
    """Describes expected configuration keys and their types/defaults."""

    required: list[str] = field(default_factory=list)
    optional: dict[str, Any] = field(default_factory=dict)  # key → default value


class ConfigManager:
    """
    Layered configuration: defaults < config file < environment variables.

    Environment variables take precedence over file, which takes precedence over defaults.
    Sensitive keys (containing 'key', 'secret', 'password', 'token') are masked in repr.
    """

    SENSITIVE_KEYWORDS = ("key", "secret", "password", "token", "private")

    def __init__(self, schema: ConfigSchema | None = None):
        self._schema = schema or ConfigSchema()
        self._data: dict[str, Any] = {}

    def load_file(self, path: str | Path) -> None:
        """Load JSON config file. Raises ConfigError if file is invalid JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}") from e
        except OSError as e:
            raise ConfigError(f"Cannot read config file: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError("Config file must be a JSON object")
        self._data.update(data)

    def load_env(self, prefix: str = "DLLM_") -> None:
        """Load environment variables with the given prefix (prefix stripped from key name)."""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                self._data[config_key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for key, or default if not set. Falls back to schema defaults."""
        if key in self._data:
            return self._data[key]
        if key in self._schema.optional:
            return self._schema.optional[key]
        return default

    def require(self, key: str) -> Any:
        """Return value or raise ConfigError if missing."""
        val = self.get(key)
        if val is None:
            raise ConfigError(f"Required config key missing: {key!r}")
        return val

    def validate(self) -> list[str]:
        """Return list of missing required keys (empty → valid)."""
        missing = []
        for key in self._schema.required:
            if self.get(key) is None:
                missing.append(key)
        return missing

    def is_sensitive(self, key: str) -> bool:
        """Return True if the key name looks sensitive."""
        lower = key.lower()
        return any(kw in lower for kw in self.SENSITIVE_KEYWORDS)

    def safe_dict(self) -> dict:
        """Return config dict with sensitive values masked as '***'."""
        result = {}
        all_keys = set(self._data) | set(self._schema.optional)
        for key in all_keys:
            value = self.get(key)
            result[key] = "***" if self.is_sensitive(key) else value
        return result

    def fingerprint(self) -> str:
        """SHA-256 hex of the current config (sorted JSON). Useful for change detection."""
        serialized = json.dumps(self._data, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()


__all__ = ["ConfigError", "ConfigManager", "ConfigSchema"]
