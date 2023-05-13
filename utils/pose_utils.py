import os
import shutil

from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from pose_format import PoseHeader, Pose
from pose_format.pose_header import PoseNormalizationInfo
from pose_format.pose_visualizer import PoseVisualizer

from utils.constants import DEFAULT_COMPONENTS


def pose_normalization_info(pose_header: PoseHeader) -> PoseNormalizationInfo:
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))

    raise ValueError("Unknown pose header schema for normalization")


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0 # Confidence Shape (Frames, People, Points)
        pose.body.data[:, :, points, :] = 0 # Data Shape (Frames, People, Points, Dims)
    else:
        raise ValueError("Unknown pose header schema for hiding legs")


def save_pose_as_video(pose_url: str,
                       video_name: str):
    if os.path.isdir('videos'):
        shutil.rmtree('videos')
    os.makedirs('videos')

    print('Loading input pose ...')
    with open(pose_url, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pose.get_components(DEFAULT_COMPONENTS)
        pose_hide_legs(pose)

        print('Generating videos ...')
        visualize_pose(pose, f'videos\\{video_name}.mp4')


def visualize_pose(pose: Pose,
                   video_name: str,
                   ffmpeg_path: str):
    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)

    visualizer.save_video(video_name, visualizer.draw(),
                          custom_ffmpeg=ffmpeg_path)


def concate_two_videos(first_video_name: str,
                       second_video_name: str,
                       final_video_name: str):
    # Load the two video files
    first_video = VideoFileClip(f'{first_video_name}.mp4')
    second_video = VideoFileClip(f'{second_video_name}.mp4')

    # Concatenate the clips
    final_video = concatenate_videoclips([first_video, second_video])

    # Write the final clip to a new file
    final_video.write_videofile(f'{final_video_name}.mp4')
