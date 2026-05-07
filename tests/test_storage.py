"""Tests for the StorageClient wrapper (no real Lighthouse API needed)."""

import pytest


class TestStorageClient:
    def test_missing_api_key_raises(self):
        """StorageClient should raise ValueError when no API key is provided."""
        from node.storage import StorageClient

        with pytest.raises(ValueError, match="LIGHTHOUSE_API_KEY"):
            StorageClient(api_key="")

    def test_missing_lighthouseweb3_raises(self, monkeypatch):
        """StorageClient should raise RuntimeError when SDK is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "lighthouseweb3":
                raise ImportError("No module named 'lighthouseweb3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from node.storage import StorageClient

        with pytest.raises(RuntimeError, match="pip install lighthouseweb3"):
            StorageClient(api_key="test-key")

    def test_gateway_list_has_multiple_entries(self):
        """Fallback gateway list should have at least 2 options."""
        from node import storage

        assert len(storage.GATEWAYS) >= 2
        for gw in storage.GATEWAYS:
            assert gw.startswith("https://")
