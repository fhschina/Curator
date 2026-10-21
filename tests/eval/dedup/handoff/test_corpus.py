from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from eval.dedup.core.config import TokenizerConfig
from eval.dedup.handoff.corpus import TokenCounter


def test_fast_length_scan_matches_transformers_without_truncation(tmp_path: Path) -> None:
    unknown = "[UNK]"
    backend = Tokenizer(WordLevel({unknown: 0, "hello": 1, "world": 2, "!": 3}, unk_token=unknown))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unknown, model_max_length=8)
    counter = TokenCounter(TokenizerConfig("whitespace", "fixture", "fixture", tmp_path))
    counter.tokenizer = tokenizer
    texts = ["", "hello world!", "中文 mixed text", "hello world " * 1000]
    expected = tokenizer(texts, add_special_tokens=False, return_length=True, truncation=False, padding=False)[
        "length"
    ]
    assert counter.count_many(texts, batch_size=2) == expected
    assert len(counter.encode_with_offsets(texts[-1])[0]) == expected[-1]
