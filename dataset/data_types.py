from dataclasses import dataclass
from typing import Optional, Dict, Union, List

from pose_format import Pose
from tensorflow.python.framework.ops import EagerTensor
import torch
from torch.utils.data import Dataset


@dataclass
class PoseItem:
    data: EagerTensor
    conf: EagerTensor
    fps: EagerTensor


@dataclass
class DataItemObject:
    gloss: EagerTensor
    hamnosys: EagerTensor
    id: EagerTensor
    signed_language: EagerTensor
    spoken_language: EagerTensor
    text: EagerTensor
    video: EagerTensor
    pose: Optional[Union[PoseItem, dict]] = None
    poses: Optional[Dict[str, PoseItem]] = None

    def __post_init__(self):
        self.pose = PoseItem(**self.pose)


@dataclass
class TextPoseDatum:
    id: str
    text: str
    pose: Pose
    length: int


class TextPoseDataset(Dataset):

    def __init__(self, data: List[TextPoseDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        pose = datum.pose

        torch_body = pose.body.torch()
        pose_length = len(torch_body.data)

        return {
            "id": datum.id,
            "text": datum.text,
            "pose": {
                "obj": pose,
                "data": torch_body.data.tensor[:, 0, :, :],
                "confidence": torch_body.confidence[:, 0, :],
                "length": torch.tensor([pose_length], dtype=torch.float),
                "inverse_mask": torch.ones(pose_length, dtype=torch.int8)
            }
        }
