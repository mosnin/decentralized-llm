"""Tests for ArweaveStorageClient (no real Arweave SDK or network needed)."""

import sys
import types
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arweave_stub():
    """Return a minimal fake arweave module."""
    stub = types.ModuleType("arweave")

    class FakeWallet:
        def __init__(self, path):
            self.path = path

    class FakeTransaction:
        def __init__(self, wallet, data=b""):
            self.wallet = wallet
            self.data = data
            self.id = "fake-tx-id-1234"
            self.tags = {}

        def add_tag(self, key, value):
            self.tags[key] = value

        def sign(self):
            pass

        def send(self):
            pass

    stub.Wallet = FakeWallet
    stub.Transaction = FakeTransaction
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArweaveStorageClientInit:
    def test_missing_arweave_package_raises(self, tmp_path):
        """RuntimeError when arweave-python-client is not installed."""
        wallet_file = tmp_path / "wallet.json"
        wallet_file.write_text("{}")

        # Remove arweave from sys.modules so the import inside __init__ fails.
        saved = sys.modules.pop("arweave", None)
        try:
            # Prevent the import from succeeding by inserting a broken finder.
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "arweave":
                    raise ImportError("No module named 'arweave'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", mock_import):
                from node.storage import ArweaveStorageClient

                with pytest.raises(RuntimeError, match="pip install arweave-python-client"):
                    ArweaveStorageClient(str(wallet_file))
        finally:
            if saved is not None:
                sys.modules["arweave"] = saved

    def test_missing_wallet_file_raises(self, tmp_path):
        """FileNotFoundError when the wallet JWK file does not exist."""
        missing = tmp_path / "does_not_exist.json"

        # Provide a stub so the ImportError branch is not hit first.
        stub = _make_arweave_stub()
        sys.modules.setdefault("arweave", stub)

        from node.storage import ArweaveStorageClient

        with pytest.raises(FileNotFoundError):
            ArweaveStorageClient(str(missing))


class TestArweaveStorageClientMethods:
    @pytest.fixture(autouse=True)
    def inject_arweave_stub(self):
        """Ensure the arweave stub is in sys.modules for every test in this class."""
        stub = _make_arweave_stub()
        original = sys.modules.get("arweave")
        sys.modules["arweave"] = stub
        yield stub
        if original is None:
            sys.modules.pop("arweave", None)
        else:
            sys.modules["arweave"] = original

    def _make_client(self, tmp_path):
        wallet_file = tmp_path / "wallet.json"
        wallet_file.write_text("{}")
        from node.storage import ArweaveStorageClient

        return ArweaveStorageClient(str(wallet_file))

    @pytest.mark.asyncio
    async def test_get_model_url_format(self, tmp_path):
        """get_model_url returns the canonical Arweave gateway URL."""
        client = self._make_client(tmp_path)
        tx_id = "abc123XYZ"
        url = await client.get_model_url(tx_id)
        assert url == f"https://arweave.net/{tx_id}"

    @pytest.mark.asyncio
    async def test_upload_creates_tar(self, tmp_path):
        """upload_model_weights archives the directory and calls transaction.send()."""
        import tarfile

        # Build a tiny model directory with one file.
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"\x00" * 16)

        # Capture the Transaction that gets constructed.
        created_transactions = []

        stub = sys.modules["arweave"]
        original_transaction_cls = stub.Transaction

        class TrackingTransaction(original_transaction_cls):
            def __init__(self, wallet, data=b""):
                super().__init__(wallet, data)
                created_transactions.append(self)

        stub.Transaction = TrackingTransaction

        try:
            client = self._make_client(tmp_path)
            tx_id = await client.upload_model_weights(model_dir, "test-model-v1")
        finally:
            stub.Transaction = original_transaction_cls

        # One transaction should have been created and sent.
        assert len(created_transactions) == 1
        txn = created_transactions[0]

        # Tags must be set correctly.
        assert txn.tags["App-Name"] == "decentralized-llm"
        assert txn.tags["Model-ID"] == "test-model-v1"
        assert txn.tags["Content-Type"] == "application/x-tar"

        # The returned ID should match.
        assert tx_id == txn.id

        # The uploaded data must be a valid tar archive containing the model file.
        import io

        with tarfile.open(fileobj=io.BytesIO(txn.data), mode="r:*") as tar:
            names = tar.getnames()
        assert any("weights.bin" in n for n in names)
