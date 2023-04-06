from dataclasses import dataclass, astuple
from typing import Tuple, Any

from data_tokenizers.base_tokenizer import BaseTokenizer


@dataclass
class ConfigPoseEncoder:
    pose_dims: Tuple[int, int] = (137, 2)
    hidden_dim: int = 128
    encoder_depth: int = 4
    encoder_heads: int = 2
    encoder_dim_feedforward: int = 2048
    max_seq_size: int = 1000
    dropout: int = 0.5

    def __iter__(self):
        return iter(astuple(self))

    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)


@dataclass
class ConfigTextEncoder:
    tokenizer: BaseTokenizer
    max_seq_size: int = 1000
    hidden_dim: int = 128
    num_layers: int = 2
    dim_feedforward: int = 2048
    encoder_heads: int = 2



