from typing import List

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseNormalizationInfo
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.reader import BufferReader
from sign_language_datasets.datasets.config import SignDatasetConfig
from tqdm import tqdm

from dataset.data_types import DataItemObject, ProcessedPoseDatum
from utils.pose_utils import pose_normalization_info, pose_hide_legs

DEFUALT_COMPONENTS = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]

def process_datum(datum: DataItemObject,
                  pose_header: PoseHeader,
                  normalization_info: PoseNormalizationInfo,
                  components: List[str] = None) -> ProcessedPoseDatum:
    tf_poses = {"": datum.pose} if datum.pose is not None else datum.poses
    poses = {}
    for key, tf_pose in tf_poses.items():
        fps = int(tf_pose.fps.numpy())
        pose_body = NumPyPoseBody(fps, tf_pose.data.numpy(), tf_pose.conf.numpy())
        pose = Pose(pose_header, pose_body)

        # # Get subset of components if needed
        # if components and len(components) != len(pose_header.components):
        #     pose = pose.get_components(components)

        pose = pose.normalize(normalization_info)
        pose_hide_legs(pose)
        poses[key] = pose

    return ProcessedPoseDatum(id=datum.id.numpy().decode('utf-8'),
                              pose=poses[""] if datum.pose is not None else poses,
                              tf_datum=datum)


def load_dataset() -> List[ProcessedPoseDatum]:
    config = SignDatasetConfig(name="cearing2", version="1.0.0", include_video=False, fps=25, include_pose="holistic")
    dicta_sign = tfds.load(name='dicta_sign', builder_kwargs={"config": config})
    with open("holistic.header", "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))
    normalization_info = pose_normalization_info(pose_header)
    return [process_datum(DataItemObject(**datum), pose_header,normalization_info,DEFUALT_COMPONENTS) for datum in tqdm(dicta_sign["train"]) if
                  datum['spoken_language'].numpy().decode('utf-8') == "en"]



def pose_visualizer(pose: Pose):
    p = PoseVisualizer(pose)
    p.save_video("results/example-video2.mp4", p.draw())


if __name__ == '__main__':
    datum = load_dataset()[0]
    pose_visualizer(datum.pose)

    # pylint: disable=protected-access

    # with open("example_video.pose", "rb") as buffer:
    #     pose_header = Pose.read(buffer.read())
    # pose_visualizer(pose_header)
    print(load_dataset()[0].tf_datum.text)
