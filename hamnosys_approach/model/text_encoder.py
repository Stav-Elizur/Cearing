from typing import List

import torch
from torch import nn

from model.model_types import ConfigTextEncoder


class TextEncoderModel(nn.Module):

    def __init__(self, config: ConfigTextEncoder):
        super().__init__()

        self.tokenizer = config.tokenizer
        self.max_seq_size = config.max_seq_size

        self.embedding = nn.Embedding(
            num_embeddings=len(config.tokenizer),
            embedding_dim=config.hidden_dim,
            padding_idx=config.tokenizer.pad_token_id,
        )

        self.positional_embedding = nn.Embedding(num_embeddings=config.max_seq_size,
                                                 embedding_dim=config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_dim,
                                                   nhead=config.encoder_heads,
                                                   dim_feedforward=config.dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Used to figure out the device of the model
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.dummy_param.device)
        positional_embedding = self.positional_embedding(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding

        encoded = self.encoder(embedding, src_key_padding_mask=tokenized["attention_mask"])

        return {"data": encoded, "mask": tokenized["attention_mask"]}
