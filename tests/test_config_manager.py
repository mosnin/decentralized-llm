import json

import pytest

from node.config_manager import ConfigError, ConfigManager, ConfigSchema


def test_get_returns_default_when_missing():
    cm = ConfigManager()
    assert cm.get("missing", "default") == "default"


def test_get_returns_schema_default():
    schema = ConfigSchema(optional={"model_name": "gpt-4"})
    cm = ConfigManager(schema=schema)
    assert cm.get("model_name") == "gpt-4"


def test_load_file_valid_json(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"host": "localhost", "port": 8080}))
    cm = ConfigManager()
    cm.load_file(cfg_file)
    assert cm.get("host") == "localhost"
    assert cm.get("port") == 8080


def test_load_file_invalid_json_raises(tmp_path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text("{not valid json}")
    cm = ConfigManager()
    with pytest.raises(ConfigError, match="Invalid JSON"):
        cm.load_file(cfg_file)


def test_load_file_missing_raises(tmp_path):
    cm = ConfigManager()
    with pytest.raises(ConfigError, match="Cannot read config file"):
        cm.load_file(tmp_path / "nonexistent.json")


def test_load_env_strips_prefix(monkeypatch):
    monkeypatch.setenv("DLLM_FOO", "bar")
    cm = ConfigManager()
    cm.load_env()
    assert cm.get("foo") == "bar"


def test_env_overrides_file(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"foo": "1"}))
    monkeypatch.setenv("DLLM_FOO", "2")
    cm = ConfigManager()
    cm.load_file(cfg_file)
    cm.load_env()
    assert cm.get("foo") == "2"


def test_require_present_key():
    cm = ConfigManager()
    cm._data["node_id"] = "abc123"
    assert cm.require("node_id") == "abc123"


def test_require_missing_raises():
    cm = ConfigManager()
    with pytest.raises(ConfigError, match="Required config key missing"):
        cm.require("missing")


def test_validate_all_present():
    schema = ConfigSchema(required=["node_id", "host"])
    cm = ConfigManager(schema=schema)
    cm._data["node_id"] = "x"
    cm._data["host"] = "localhost"
    assert cm.validate() == []


def test_validate_missing_required():
    schema = ConfigSchema(required=["node_id", "host"])
    cm = ConfigManager(schema=schema)
    cm._data["host"] = "localhost"
    missing = cm.validate()
    assert missing == ["node_id"]


def test_is_sensitive_api_key():
    cm = ConfigManager()
    assert cm.is_sensitive("api_key") is True


def test_is_sensitive_normal_key():
    cm = ConfigManager()
    assert cm.is_sensitive("model_name") is False


def test_safe_dict_masks_sensitive():
    cm = ConfigManager()
    cm._data["api_key"] = "super-secret-value"
    result = cm.safe_dict()
    assert result["api_key"] == "***"


def test_safe_dict_shows_non_sensitive():
    cm = ConfigManager()
    cm._data["model_name"] = "llama-3"
    result = cm.safe_dict()
    assert result["model_name"] == "llama-3"


def test_fingerprint_changes_with_data():
    cm1 = ConfigManager()
    cm1._data["x"] = "1"
    cm2 = ConfigManager()
    cm2._data["x"] = "2"
    assert cm1.fingerprint() != cm2.fingerprint()
