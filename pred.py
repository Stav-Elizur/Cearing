import os
import shutil
from typing import List

import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.pose_visualizer import PoseVisualizer

from data_tokenizers.hamnosys_tokenizer import HamNoSysTokenizer
from dataset.data import load_dataset
from model.Iterative_decoder import IterativeGuidedPoseGenerationModel
from model.model_types import ConfigPoseEncoder, ConfigTextEncoder
from model.pose_encoder import PoseEncoderModel
from utils.constants import MAX_SEQ_SIZE, DEFAULT_COMPONENTS, DATA_DIR
from utils.pose_utils import pose_normalization_info, pose_hide_legs
from model.text_encoder import TextEncoderModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU

VIDEOS_PATH = "videos"


def visualize_pose(pose: Pose, pose_name: str):
    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()

    if pose.header.dimensions.height % 2 == 1:
        pose.header.dimensions.height += 1

    if pose.header.dimensions.width % 2 == 1:
        pose.header.dimensions.width += 1

    if pose.header.dimensions.depth % 2 == 1:
        pose.header.dimensions.depth += 1

    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)
    visualizer.save_video(os.path.join(VIDEOS_PATH, pose_name), visualizer.draw(),
                          custom_ffmpeg="C:\projects\\ffmpeg\\bin")

def visualize_poses(_id: str, text: str, poses: List[Pose]) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3> (original / pred / pred + length / cfg) <br />"

    for k, pose in enumerate(poses):
        pose_name = f"{_id}_{k}.mp4"
        visualize_pose(pose, pose_name)
        html_tags += f"<video src=\"{pose_name}\" controls preload='none'></video>"

    return html_tags


def data_to_pose(pred_seq, pose_header: PoseHeader):
    data = list(pred_seq)[-1]
    data = torch.unsqueeze(data, 1).cpu()
    conf = torch.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25 if None is None else 45, data.numpy(), conf.numpy())
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    return predicted_pose


if __name__ == '__main__':

    train_dataset = load_dataset(split="train[:1%]",
                                 max_seq_size=MAX_SEQ_SIZE,
                                 components=DEFAULT_COMPONENTS,
                                 data_dir=DATA_DIR)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape
    pose_header = train_dataset.data[0].pose.header

    pose_encoder = PoseEncoderModel(ConfigPoseEncoder(pose_dims=(num_pose_joints, num_pose_dims),
                                                      dropout=0,
                                                      max_seq_size=MAX_SEQ_SIZE))

    text_encoder = TextEncoderModel(ConfigTextEncoder(tokenizer=HamNoSysTokenizer(),
                                                      max_seq_size=MAX_SEQ_SIZE))

    # Model Arguments
    model_args = dict(pose_encoder=pose_encoder,
                      text_encoder=text_encoder,
                      hidden_dim=128,
                      learning_rate=1e-4,
                      seq_len_loss_weight=2e-5,
                      smoothness_loss_weight=1e-2,
                      noise_epsilon=1e-3,
                      num_steps=100)

    model = IterativeGuidedPoseGenerationModel.load_from_checkpoint('models/1/model-v1.ckpt', **model_args)
    model.eval()

    html = []

    with torch.no_grad():
        for datum in train_dataset:
            pose_data = datum["pose"]["data"]
            first_pose = pose_data[0]
            sequence_length = pose_data.shape[0]
            # datum["text"] = ""
            pred_normal = model.forward(text=datum["text"], first_pose=first_pose)
            pred_len = model.forward(text=datum["text"], first_pose=first_pose, force_sequence_length=sequence_length)
            pred_cfg = model.forward(text=datum["text"], first_pose=first_pose, classifier_free_guidance=2.5)

            html.append(
                visualize_poses(_id=datum["id"],
                                text=datum["text"],
                                poses=[
                                    datum["pose"]["obj"],
                                    data_to_pose(pred_normal, pose_header),
                                    data_to_pose(pred_len, pose_header),
                                    data_to_pose(pred_cfg, pose_header)
                                ]))

    with open(os.path.join("videos", "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
        f.write("<br><br><br>".join(html))

    shutil.copyfile(text_encoder.tokenizer.font_path, os.path.join("videos", "HamNoSys.ttf"))
