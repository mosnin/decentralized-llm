import hashlib

import pytest

from node.merkle import MerkleProof, MerkleTree, _combine, _hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# ---------------------------------------------------------------------------
# Root computation tests
# ---------------------------------------------------------------------------


def test_single_leaf_root():
    leaf = b"only leaf"
    tree = MerkleTree([leaf])
    assert tree.root == sha256(leaf)


def test_two_leaves_root():
    a = b"leaf a"
    b = b"leaf b"
    tree = MerkleTree([a, b])
    expected = sha256(sha256(a) + sha256(b))
    assert tree.root == expected


# ---------------------------------------------------------------------------
# Proof verification tests
# ---------------------------------------------------------------------------


def test_proof_verifies_for_first_leaf():
    leaves = [b"alpha", b"beta", b"gamma", b"delta"]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(0)
    assert proof.verify() is True


def test_proof_verifies_for_last_leaf():
    leaves = [b"one", b"two", b"three", b"four"]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(len(leaves) - 1)
    assert proof.verify() is True


def test_proof_fails_tampered_leaf():
    leaves = [b"data0", b"data1", b"data2", b"data3"]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(1)
    # Tamper the leaf hash
    tampered = MerkleProof(
        leaf_index=proof.leaf_index,
        leaf_hash=b"\x00" * 32,
        siblings=proof.siblings,
        root=proof.root,
    )
    assert tampered.verify() is False


def test_proof_fails_tampered_root():
    leaves = [b"x", b"y", b"z", b"w"]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(2)
    tampered = MerkleProof(
        leaf_index=proof.leaf_index,
        leaf_hash=proof.leaf_hash,
        siblings=proof.siblings,
        root=b"\xff" * 32,
    )
    assert tampered.verify() is False


def test_proof_fails_tampered_sibling():
    leaves = [b"node0", b"node1", b"node2", b"node3"]
    tree = MerkleTree(leaves)
    proof = tree.get_proof(0)
    # Replace first sibling hash with zeroes
    bad_siblings = [(b"\x00" * 32, proof.siblings[0][1])] + list(proof.siblings[1:])
    tampered = MerkleProof(
        leaf_index=proof.leaf_index,
        leaf_hash=proof.leaf_hash,
        siblings=bad_siblings,
        root=proof.root,
    )
    assert tampered.verify() is False


# ---------------------------------------------------------------------------
# Bulk correctness
# ---------------------------------------------------------------------------


def test_all_leaves_prove_correctly():
    leaves = [f"leaf{i}".encode() for i in range(8)]
    tree = MerkleTree(leaves)
    for i in range(8):
        assert tree.get_proof(i).verify() is True, f"Proof failed for leaf {i}"


# ---------------------------------------------------------------------------
# Padding / edge cases
# ---------------------------------------------------------------------------


def test_odd_leaf_count_padded():
    """3 leaves get padded to 4; all original proofs must still verify."""
    leaves = [b"a", b"b", b"c"]
    tree = MerkleTree(leaves)
    # Tree should have 4 padded leaves internally
    assert len(tree._leaves) == 4
    # All original 3 leaves verify
    for i in range(3):
        assert tree.get_proof(i).verify() is True


def test_index_out_of_range_raises():
    leaves = [b"hello", b"world"]
    tree = MerkleTree(leaves)
    with pytest.raises(IndexError):
        tree.get_proof(100)


def test_empty_leaves_raises():
    with pytest.raises(ValueError):
        MerkleTree([])


# ---------------------------------------------------------------------------
# Internal consistency: _hash and _combine
# ---------------------------------------------------------------------------


def test_hash_is_sha256():
    data = b"test data"
    assert _hash(data) == sha256(data)


def test_combine_is_hash_of_concatenation():
    left = b"L" * 32
    right = b"R" * 32
    assert _combine(left, right) == sha256(left + right)
