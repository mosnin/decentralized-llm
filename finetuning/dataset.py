"""
Local dataset wrapper — training data stays on the node, never shared.

Supports:
  - JSONL files: {"text": "..."} or {"prompt": "...", "completion": "..."}
  - Plain text files (one document per line)
  - HuggingFace dataset names (downloaded locally)
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class LocalDataset(Dataset):
    """
    Loads text data from local files and tokenizes it for causal LM training.
    Data never leaves this node — only gradients are shared with peers.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load(Path(data_path))

    def _load(self, path: Path) -> list[str]:
        texts = []
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    texts.append(obj["text"])
                elif "prompt" in obj and "completion" in obj:
                    texts.append(obj["prompt"] + obj["completion"])
        elif path.suffix in (".txt", ".md"):
            texts = [ln for ln in path.read_text().splitlines() if ln.strip()]
        else:
            raise ValueError(f"Unsupported data format: {path.suffix}. Use .jsonl or .txt")
        return texts

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
