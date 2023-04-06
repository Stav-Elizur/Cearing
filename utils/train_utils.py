from typing import Union, Dict, Tuple, List

import numpy as np
import torch
from pose_format.torch.masked import MaskedTorch, MaskedTensor

from dataset.data_types import TextPoseItem


def collate_tensors(batch: List, pad_value=0) -> Union[torch.Tensor, List]:
    datum = batch[0]

    if isinstance(datum, dict):  # Recurse over dictionaries
        return zero_pad_collator(batch)

    if isinstance(datum, (int, np.int32)):
        return torch.tensor(batch, dtype=torch.long)

    if isinstance(datum, (MaskedTensor, torch.Tensor)):
        max_len = max(len(t) for t in batch)
        if max_len == 1:
            return torch.stack(batch)

        torch_cls = MaskedTorch if isinstance(datum, MaskedTensor) else torch

        new_batch = []
        for tensor in batch:
            missing = list(tensor.shape)
            missing[0] = max_len - tensor.shape[0]

            if missing[0] > 0:
                padding_tensor = torch.full(missing, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch_cls.cat([tensor, padding_tensor], dim=0)

            new_batch.append(tensor)

        return torch_cls.stack(new_batch, dim=0)

    return batch


def zero_pad_collator(batch) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
    datum = batch[0]

    # For strings
    if isinstance(datum, str):
        return batch

    # For tuples
    if isinstance(datum, tuple):
        return tuple(collate_tensors([b[i] for b in batch]) for i in range(len(datum)))

    # For classes
    if isinstance(datum, TextPoseItem):
        keys = datum.__dict__.keys()
        return {k: collate_tensors([b.__dict__[k] for b in batch]) for k in keys}

    # For dictionaries
    keys = datum.keys()
    return {k: collate_tensors([b[k] for b in batch]) for k in keys}