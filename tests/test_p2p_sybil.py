"""
Tests for S/Kademlia sybil-resistance mechanisms in node/p2p.py.

All tests run without hivemind installed: the module is imported after
stubbing out the `hivemind` and `torch` packages in sys.modules so that
the sybil-resistance helpers (which are pure Python) can be exercised in
isolation.
"""

import hashlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so the module loads without them
# ---------------------------------------------------------------------------


def _install_stubs():
    """Install minimal sys.modules stubs for hivemind and torch."""
    for name in ("hivemind", "torch"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    torch_stub = sys.modules["torch"]

    # torch.Tensor stub — used as a type annotation in ShardExpert.forward()
    if not hasattr(torch_stub, "Tensor"):
        torch_stub.Tensor = object  # type: ignore[attr-defined]

    # torch.nn.Module stub — ShardExpert inherits from it
    if not hasattr(torch_stub, "nn"):
        nn_stub = types.ModuleType("torch.nn")
        sys.modules["torch.nn"] = nn_stub

        class _Module:
            def __init__(self):
                pass

        nn_stub.Module = _Module  # type: ignore[attr-defined]
        torch_stub.nn = nn_stub  # type: ignore[attr-defined]


_install_stubs()

# Now we can safely import our module
from node.p2p import SybilResistantDHT  # noqa: E402  (import after stub setup)

# ---------------------------------------------------------------------------
# Helper: build a legitimate node ID from a public key
# ---------------------------------------------------------------------------


def _make_keypair_and_id() -> tuple[bytes, str]:
    """Return (pubkey_bytes, valid_node_id) pair."""
    pubkey_bytes = b"test-public-key-material-32bytes"
    node_id = hashlib.sha256(pubkey_bytes).hexdigest()
    return pubkey_bytes, node_id


# ---------------------------------------------------------------------------
# 1. verify_node_id: accepts a legitimately derived ID
# ---------------------------------------------------------------------------


class TestVerifyNodeId:
    def test_node_id_derived_from_pubkey(self):
        """verify_node_id must return True when the ID is SHA-256(pubkey)."""
        pubkey_bytes, valid_id = _make_keypair_and_id()
        assert SybilResistantDHT.verify_node_id(pubkey_bytes, valid_id) is True

    def test_node_id_mismatch_rejected(self):
        """verify_node_id must return False when the ID does not match the pubkey."""
        pubkey_bytes, _ = _make_keypair_and_id()
        # Use a completely different public key to generate a mismatched ID
        other_pubkey = b"a-completely-different-pubkey-xx"
        mismatched_id = hashlib.sha256(other_pubkey).hexdigest()
        assert SybilResistantDHT.verify_node_id(pubkey_bytes, mismatched_id) is False

    def test_partial_id_prefix_match(self):
        """Only the first 40 hex characters are compared (160-bit prefix)."""
        pubkey_bytes, valid_id = _make_keypair_and_id()
        # Supply only first 40 chars — should still pass
        short_id = valid_id[:40]
        assert SybilResistantDHT.verify_node_id(pubkey_bytes, short_id) is True

    def test_empty_claimed_id_rejected(self):
        """An empty claimed_id should not accidentally match."""
        pubkey_bytes = b"any-key"
        # An empty string[:40] is "", SHA-256 hex[:40] is 40 chars — never equal
        assert SybilResistantDHT.verify_node_id(pubkey_bytes, "") is False


# ---------------------------------------------------------------------------
# 2. lookup_with_redundancy: multiple disjoint paths queried
# ---------------------------------------------------------------------------


class TestRedundantLookup:
    """Tests for P2PLayer.lookup_with_redundancy using a mocked DHT."""

    def _make_p2p_layer_with_mock_dht(self, dht_get_side_effect=None):
        """
        Build a P2PLayer-like object whose DHT is mocked.

        We bypass __init__ (which requires hivemind) and inject a mock DHT
        directly.
        """
        from node.p2p import P2PLayer

        layer = object.__new__(P2PLayer)
        mock_dht = MagicMock()
        if dht_get_side_effect is not None:
            mock_dht.get.side_effect = dht_get_side_effect
        else:
            mock_dht.get.return_value = {"endpoint": "127.0.0.1:5000"}
        layer.dht = mock_dht
        return layer, mock_dht

    @pytest.mark.asyncio
    async def test_redundant_lookup_aggregates_results(self):
        """
        lookup_with_redundancy(key, k=4) must issue ≥4 DHT get() calls and
        return a non-empty merged list.
        """
        layer, mock_dht = self._make_p2p_layer_with_mock_dht()

        results = await layer.lookup_with_redundancy("test-key", k=4)

        # At least 4 calls were made (one per path, possibly more for fallback)
        assert mock_dht.get.call_count >= 4
        # At least one result was returned
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_redundant_lookup_deduplicates(self):
        """
        Identical results from different paths appear only once in the output.
        """
        # All paths return the same value
        layer, mock_dht = self._make_p2p_layer_with_mock_dht(
            dht_get_side_effect=lambda key: {"endpoint": "same-node:9999"}
        )

        results = await layer.lookup_with_redundancy("dup-key", k=6)
        # Results are deduplicated by repr
        assert results.count({"endpoint": "same-node:9999"}) == 1

    @pytest.mark.asyncio
    async def test_redundant_lookup_handles_partial_failures(self):
        """
        If some paths raise exceptions, lookup_with_redundancy still returns
        results from the successful paths.
        """
        call_count = 0

        def intermittent(key):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ConnectionError("simulated network failure")
            return {"endpoint": f"node-{call_count}:8000"}

        layer, mock_dht = self._make_p2p_layer_with_mock_dht(dht_get_side_effect=intermittent)

        # Should not raise even though half the paths fail
        results = await layer.lookup_with_redundancy("partial-key", k=6)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_redundant_lookup_raises_when_dht_not_started(self):
        """lookup_with_redundancy raises RuntimeError if DHT is None."""
        from node.p2p import P2PLayer

        layer = object.__new__(P2PLayer)
        layer.dht = None

        with pytest.raises(RuntimeError, match="DHT is not started"):
            await layer.lookup_with_redundancy("any-key")


# ---------------------------------------------------------------------------
# 3. Sibling list / eclipse detection
# ---------------------------------------------------------------------------


class TestSiblingList:
    """Tests for the sibling list and eclipse-attack detection."""

    def _make_srd(self, node_id: str, k: int = 20) -> SybilResistantDHT:
        return SybilResistantDHT(node_id=node_id, _sibling_k=k)

    def test_sibling_list_populated_and_trimmed(self):
        """add_to_sibling_list keeps at most _sibling_k entries."""
        srd = self._make_srd("a" * 40, k=5)
        for i in range(10):
            srd.add_to_sibling_list(hex(i)[2:].zfill(40))
        assert len(srd.sibling_list) <= 5

    def test_sibling_list_no_duplicates(self):
        """Adding the same node ID twice does not create duplicates."""
        srd = self._make_srd("b" * 40, k=20)
        nid = "c" * 40
        srd.add_to_sibling_list(nid)
        srd.add_to_sibling_list(nid)
        assert srd.sibling_list.count(nid) == 1

    def test_sibling_list_detects_eclipse(self):
        """
        Eclipse detection: with 20 fake nodes all closer than any real node and
        none of them in the known-nodes set, detect_eclipse returns True.
        """
        my_id = "0" * 40  # Our node ID: all zeros
        srd = self._make_srd(my_id, k=20)

        # Populate sibling list entirely with attacker-controlled IDs
        fake_ids = [hex(i)[2:].zfill(40) for i in range(1, 21)]
        for fid in fake_ids:
            srd.add_to_sibling_list(fid)

        # The set of legitimately known nodes contains none of the fakes
        known_nodes: set[str] = {"f" * 40, "e" * 40}  # two real but distant nodes

        assert srd.detect_eclipse(known_nodes) is True

    def test_no_eclipse_when_most_siblings_known(self):
        """
        No eclipse is flagged when the majority of siblings are in the
        known-nodes set (healthy, well-connected routing table).
        """
        my_id = "0" * 40
        srd = self._make_srd(my_id, k=10)

        # 8 known nodes + 2 unknown → 20 % unknown, below 50 % threshold
        known_nodes: set[str] = set()
        for i in range(1, 9):
            nid = hex(i)[2:].zfill(40)
            srd.add_to_sibling_list(nid)
            known_nodes.add(nid)
        for i in range(9, 11):
            srd.add_to_sibling_list(hex(i)[2:].zfill(40))

        assert srd.detect_eclipse(known_nodes) is False

    def test_detect_eclipse_empty_sibling_list(self):
        """detect_eclipse returns False (safe default) on an empty sibling list."""
        srd = self._make_srd("d" * 40, k=20)
        assert srd.detect_eclipse(set()) is False
