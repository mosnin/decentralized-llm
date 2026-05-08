"""
Paged KV-cache inspired by vLLM's PagedAttention.

Memory is divided into fixed-size blocks (pages).  Each sequence owns a
logical block table that maps logical page indices to physical page slots.
This eliminates fragmentation and allows prefix pages to be shared across
sequences via copy-on-write (CoW).

All torch imports are lazy (inside methods) so this module is importable
without torch installed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PagedKVCache:
    """
    Paged KV-cache inspired by vLLM's PagedAttention.

    Memory is divided into fixed-size blocks (pages). Each sequence gets a
    logical block table mapping logical page indices to physical page slots.
    This eliminates fragmentation and allows sharing of prefix pages across
    sequences (copy-on-write).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 512,
        dtype=None,
        device: str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device

        import torch as _torch  # noqa: PLC0415

        if dtype is None:
            dtype = _torch.float32

        # Physical KV store: [num_layers, 2, max_blocks, block_size, num_heads, head_dim]
        # Dimension 1: 0 = keys, 1 = values
        self._store: torch.Tensor = _torch.zeros(
            (num_layers, 2, max_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Set of free physical block indices
        self._free_blocks: set[int] = set(range(max_blocks))

        # seq_id -> list of physical block indices (block table)
        self._block_tables: dict[int, list[int]] = {}

        # seq_id -> reference count per block (physical block idx -> ref count)
        # Used for copy-on-write: a block shared by N sequences has ref_count N.
        self._block_refcounts: dict[int, int] = defaultdict(int)

        # seq_id -> number of tokens written so far
        self._seq_fill: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Reserve physical blocks for a new sequence. Returns the block table."""
        if seq_id in self._block_tables:
            raise ValueError(f"seq_id {seq_id} is already allocated")

        num_blocks = math.ceil(num_tokens / self.block_size)
        if num_blocks > len(self._free_blocks):
            raise MemoryError(
                f"Not enough free blocks: need {num_blocks}, have {len(self._free_blocks)}"
            )

        blocks = []
        for _ in range(num_blocks):
            blk = self._free_blocks.pop()
            blocks.append(blk)
            self._block_refcounts[blk] += 1

        self._block_tables[seq_id] = blocks
        self._seq_fill[seq_id] = 0
        return list(blocks)

    def free(self, seq_id: int) -> None:
        """Release all blocks for a completed sequence."""
        if seq_id not in self._block_tables:
            return

        for blk in self._block_tables[seq_id]:
            self._block_refcounts[blk] -= 1
            if self._block_refcounts[blk] == 0:
                self._free_blocks.add(blk)
                del self._block_refcounts[blk]

        del self._block_tables[seq_id]
        del self._seq_fill[seq_id]

    def write(
        self,
        seq_id: int,
        layer_idx: int,
        token_pos: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write a K/V pair at the given token position for this sequence."""
        if seq_id not in self._block_tables:
            raise KeyError(f"seq_id {seq_id} not allocated")

        logical_block = token_pos // self.block_size
        block_offset = token_pos % self.block_size
        block_table = self._block_tables[seq_id]

        if logical_block >= len(block_table):
            raise IndexError(f"token_pos {token_pos} exceeds allocated blocks for seq {seq_id}")

        physical_block = block_table[logical_block]

        # Copy-on-write: if this block is shared with another sequence, clone it
        if self._block_refcounts[physical_block] > 1:
            physical_block = self._cow_clone(seq_id, logical_block)

        self._store[layer_idx, 0, physical_block, block_offset] = key
        self._store[layer_idx, 1, physical_block, block_offset] = value

        # Advance fill pointer
        new_fill = token_pos + 1
        if new_fill > self._seq_fill[seq_id]:
            self._seq_fill[seq_id] = new_fill

    def read(self, seq_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all cached K/V tensors for a sequence up to the current fill level."""
        import torch as _torch  # noqa: PLC0415

        if seq_id not in self._block_tables:
            raise KeyError(f"seq_id {seq_id} not allocated")

        fill = self._seq_fill[seq_id]
        block_table = self._block_tables[seq_id]

        # Gather physical blocks in order and concatenate along the token axis
        chunks_k: list[torch.Tensor] = []
        chunks_v: list[torch.Tensor] = []
        remaining = fill

        for blk in block_table:
            if remaining <= 0:
                break
            tokens_in_block = min(remaining, self.block_size)
            chunks_k.append(self._store[layer_idx, 0, blk, :tokens_in_block])
            chunks_v.append(self._store[layer_idx, 1, blk, :tokens_in_block])
            remaining -= tokens_in_block

        if not chunks_k:
            empty = _torch.zeros(
                (0, self.num_heads, self.head_dim),
                dtype=self._store.dtype,
                device=self._store.device,
            )
            return empty, empty.clone()

        return _torch.cat(chunks_k, dim=0), _torch.cat(chunks_v, dim=0)

    def share_prefix(self, src_seq_id: int, dst_seq_id: int, num_shared_tokens: int) -> list[int]:
        """
        Create a new sequence (dst) that shares the first `num_shared_tokens`
        prefix blocks of src via copy-on-write.

        The dst sequence is allocated with the shared prefix already in place.
        Any subsequent write to a shared block triggers a CoW clone so that
        src and dst remain independent.

        Returns the block table for the new sequence.
        """
        if src_seq_id not in self._block_tables:
            raise KeyError(f"src_seq_id {src_seq_id} not allocated")
        if dst_seq_id in self._block_tables:
            raise ValueError(f"dst_seq_id {dst_seq_id} is already allocated")
        if num_shared_tokens < 0:
            raise ValueError("num_shared_tokens must be non-negative")

        src_blocks = self._block_tables[src_seq_id]
        num_shared_blocks = math.ceil(num_shared_tokens / self.block_size)
        if num_shared_blocks > len(src_blocks):
            raise ValueError(f"num_shared_tokens {num_shared_tokens} exceeds src allocation")

        # Point dst at the same physical blocks (increment refcounts)
        shared = src_blocks[:num_shared_blocks]
        for blk in shared:
            self._block_refcounts[blk] += 1

        self._block_tables[dst_seq_id] = list(shared)
        self._seq_fill[dst_seq_id] = num_shared_tokens
        return list(shared)

    def utilization(self) -> float:
        """Return fraction of physical blocks currently allocated (0.0–1.0)."""
        used = self.max_blocks - len(self._free_blocks)
        return used / self.max_blocks

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _cow_clone(self, seq_id: int, logical_block: int) -> int:
        """
        Clone a shared physical block for seq_id at logical_block position.
        Decrements refcount on the old block, allocates a new one, copies data.
        Returns the new physical block index.
        """
        if not self._free_blocks:
            raise MemoryError("No free blocks available for copy-on-write clone")

        old_physical = self._block_tables[seq_id][logical_block]
        new_physical = self._free_blocks.pop()

        # Copy all layers at once
        self._store[:, :, new_physical, :] = self._store[:, :, old_physical, :]

        # Update refcounts
        self._block_refcounts[old_physical] -= 1
        if self._block_refcounts[old_physical] == 0:
            self._free_blocks.add(old_physical)
            del self._block_refcounts[old_physical]

        self._block_refcounts[new_physical] += 1
        self._block_tables[seq_id][logical_block] = new_physical

        return new_physical
