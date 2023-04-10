import numpy as np
import torch
from torch import nn

from model.model_types import ConfigPoseEncoder


class PoseEncoderModel(nn.Module):

    def __init__(self, config: ConfigPoseEncoder):
        super().__init__()
        pose_dims, hidden_dim, encoder_depth, encoder_heads, encoder_dim_feedforward, max_seq_size, dropout = config

        self.dropout = nn.Dropout(p=dropout)

        self.max_seq_size = max_seq_size
        self.pose_dims = pose_dims
        self.pose_dim = int(np.prod(pose_dims))

        # Embedding layers
        self.positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)

        self.pose_projection = nn.Linear(self.pose_dim, hidden_dim)

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                        nhead=encoder_heads,
                                                        dim_feedforward=encoder_dim_feedforward,
                                                        batch_first=True)
        self.pose_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_depth)

    def forward(self, pose, additional_sequence=None):
        """

        :param pose: Dictionary including "data" (torch.Tenosr: Batch, Length, Points, Dimensions) and
                     "mask" (torch.BoolTensor: Batch, Length)
        :param additional_sequence: Dictionary including "data" (torch.Tenosr: Batch, Length, Embedding) and
                     "mask" (torch.BoolTensor: Batch, Length)
        :return: torch.Tensor
        """
        # Repeat the first frame for initial prediction
        batch_size, seq_length, _, _ = pose["data"].shape

        pose_data = self.dropout(pose["data"])

        flat_pose_data = pose_data.reshape(batch_size, seq_length, -1)

        positions = torch.arange(0, seq_length, dtype=torch.int, device=pose_data.device)
        positional_embedding = self.positional_embeddings(positions)

        # Encode pose sequence
        embedding = self.pose_projection(flat_pose_data) + positional_embedding
        mask = pose["mask"]

        if additional_sequence is not None:
            embedding = torch.cat([embedding, additional_sequence["data"]], dim=1)
            mask = torch.cat([mask, additional_sequence["mask"]], dim=1)

        return self.pose_encoder(embedding, src_key_padding_mask=mask)
