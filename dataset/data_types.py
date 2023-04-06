from dataclasses import dataclass
from typing import Optional, Dict, Union

from pose_format import Pose
from tensorflow.python.framework.ops import EagerTensor

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
    pose: Optional[Union[PoseItem,dict]] = None
    poses: Optional[Dict[str,PoseItem]] =None

    def __post_init__(self):
        self.pose = PoseItem(**self.pose)

@dataclass
class ProcessedPoseDatum:
    id: str
    pose: Union[Pose, Dict[str, Pose]]
    tf_datum: DataItemObject

@dataclass
class TextPoseDatum:
    id: str
    text: str
    pose: Pose
    length: int