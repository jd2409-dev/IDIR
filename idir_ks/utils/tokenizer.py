"""Tokenizer Module for IDIR-KS

Supports SentencePiece BPE tokenization with fallback to character-level.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union


class IDIRSTokenizer:
    """Wrapper around SentencePiece tokenizer with fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 8192,
        fallback: bool = True,
    ):
        self.vocab_size = vocab_size
        self.sp = None
        self.use_fallback = fallback
        self.char_to_id = {}
        self.id_to_char = {}

        if model_path and os.path.exists(model_path):
            self._load_sentencepiece(model_path)
        else:
            self._build_char_vocab()

    def _load_sentencepiece(self, model_path: str):
        """Load trained SentencePiece model."""
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.vocab_size()
        self.use_fallback = False

    def _build_char_vocab(self):
        """Build character-level vocabulary as fallback."""
        chars = (
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            + "!@#$%^&*()-_=+[]{}|;:',.<>?/\\\"~`"
            + "\n\t\r"
        )
        for i, c in enumerate(chars):
            if i < self.vocab_size - 1:
                self.char_to_id[c] = i + 1
                self.id_to_char[i + 1] = c
        self.char_to_id["<unk>"] = 0
        self.id_to_char[0] = "<unk>"

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is not None:
            ids = self.sp.encode(text, out_type=int)
            if add_bos:
                ids = [self.sp.bos_id()] + ids
            if add_eos:
                ids = ids + [self.sp.eos_id()]
            return ids
        else:
            ids = [self.char_to_id.get(c, 0) for c in text]
            if add_bos:
                ids = [1] + ids
            if add_eos:
                ids = ids + [2]
            return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if self.sp is not None:
            if skip_special:
                ids = [
                    i
                    for i in ids
                    if i
                    not in (
                        self.sp.bos_id(),
                        self.sp.eos_id(),
                        self.sp.unk_id(),
                        self.sp.pad_id(),
                    )
                ]
            return self.sp.decode(ids)
        else:
            return "".join(self.id_to_char.get(i, "") for i in ids)

    def __call__(
        self, text: str, max_length: int = None, truncation: bool = False, **kwargs
    ) -> Dict:
        """Callable interface compatible with HuggingFace-style tokenizers."""
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if max_length is not None and not truncation:
            ids = ids + [0] * (max_length - len(ids))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    @property
    def bos_token_id(self) -> int:
        if self.sp:
            return self.sp.bos_id()
        return 1

    @property
    def eos_token_id(self) -> int:
        if self.sp:
            return self.sp.eos_id()
        return 2

    @property
    def pad_token_id(self) -> int:
        if self.sp:
            return self.sp.pad_id()
        return 0

    @property
    def unk_token_id(self) -> int:
        if self.sp:
            return self.sp.unk_id()
        return 0


def train_tokenizer(
    input_files: List[str],
    model_path: str,
    vocab_size: int = 8192,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
):
    """
    Train a SentencePiece tokenizer on text files.

    Args:
        input_files: Paths to text files for training
        model_path: Where to save the trained model (without extension)
        vocab_size: Vocabulary size
        model_type: 'bpe', 'unigram', 'char', or 'word'
        character_coverage: Coverage of characters (0.9995 for English, 1.0 for Japanese/Chinese)
    """
    import sentencepiece as spm

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        max_sentence_length=1000000,
        shuffle_input_sentence=True,
        seed=42,
        normalization_rule_name="nmt_nfkc",
        byte_fallback=True,
        unk_surface=" \u2047 ",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
    )

    print(f"Tokenizer trained and saved to {model_path}.model / {model_path}.vocab")
    return IDIRSTokenizer(model_path=f"{model_path}.model")
