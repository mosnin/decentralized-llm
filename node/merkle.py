import hashlib
from dataclasses import dataclass


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _combine(left: bytes, right: bytes) -> bytes:
    return _hash(left + right)


@dataclass
class MerkleProof:
    leaf_index: int
    leaf_hash: bytes
    siblings: list[tuple[bytes, str]]  # (hash, "left"|"right") — direction of sibling
    root: bytes

    def verify(self) -> bool:
        """Recompute root from leaf + siblings and compare to self.root."""
        current = self.leaf_hash
        for sibling_hash, direction in self.siblings:
            if direction == "left":
                current = _combine(sibling_hash, current)
            else:
                current = _combine(current, sibling_hash)
        return current == self.root


class MerkleTree:
    """
    Binary Merkle tree over a list of byte-string leaves.
    Pads to next power of 2 by duplicating the last leaf if needed.
    """

    def __init__(self, leaves: list[bytes]):
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        # Pad to next power of 2
        n = 1
        while n < len(leaves):
            n <<= 1
        self._leaves = leaves + [leaves[-1]] * (n - len(leaves))
        self._tree = self._build()

    def _build(self) -> list[list[bytes]]:
        """Build tree as list of levels, bottom-up. Level 0 = leaf hashes."""
        levels = [[_hash(leaf) for leaf in self._leaves]]
        while len(levels[-1]) > 1:
            prev = levels[-1]
            levels.append([_combine(prev[i], prev[i + 1]) for i in range(0, len(prev), 2)])
        return levels

    @property
    def root(self) -> bytes:
        return self._tree[-1][0]

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """Return a Merkle proof for the leaf at leaf_index (0-based, unpadded)."""
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range")

        leaf_hash = self._tree[0][leaf_index]
        siblings = []
        idx = leaf_index

        for level in self._tree[:-1]:  # all levels except root
            if idx % 2 == 0:
                # idx is left child; sibling is idx+1 (or same if last)
                sib = level[min(idx + 1, len(level) - 1)]
                siblings.append((sib, "right"))
            else:
                # idx is right child; sibling is idx-1
                siblings.append((level[idx - 1], "left"))
            idx //= 2

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=leaf_hash,
            siblings=siblings,
            root=self.root,
        )
