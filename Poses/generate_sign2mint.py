import json
import subprocess
import zipfile
import cv2
import os
import shutil

from pose_format import Pose
from pose_format.pose_header import PoseNormalizationInfo, PoseHeader
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.holistic import load_holistic
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
from tqdm import tqdm


def load_video_frames(video_name: str):
    print('Loading video ...')
    cap = cv2.VideoCapture(f'{video_name}.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def generate_pose(video_name: str):
    cap = cv2.VideoCapture(f'{video_name}.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    frames = load_video_frames(video_name)
    print('Estimating pose ...')
    return load_holistic(frames,
                         fps=fps,
                         width=width,
                         height=height,
                         progress=True,
                         additional_holistic_config={'model_complexity': 2})


def save_pose(video_name: str, pose_name: str):
    # Perform pose estimation
    pose = generate_pose(video_name)

    # Write
    print('Saving to disk ...')
    with open(f'{pose_name}.pose', "wb") as f:
        pose.write(f)


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


def visualize_pose(pose: Pose, pose_name: str):
    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)

    visualizer.save_video(pose_name, visualizer.draw(),
                          custom_ffmpeg="/usr/bin/ffmpeg")


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


def save_video(pose_name: str, video_name: str):
    DEFAULT_COMPONENTS = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "FACE_LANDMARKS"]

    print('Loading input pose ...')
    with open(f'{pose_name}.pose', 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = pose.get_components(DEFAULT_COMPONENTS)
        pose_hide_legs(pose)

        print('Generating videos ...')
        visualize_pose(pose, f'{video_name}.mp4')


def decode_surrogates(string):
    result = ""
    i = 0
    while i < len(string):
        if 0xD800 <= ord(string[i]) <= 0xDBFF and i + 1 < len(string) and 0xDC00 <= ord(string[i + 1]) <= 0xDFFF:
            high_surrogate = ord(string[i])
            low_surrogate = ord(string[i + 1])
            code_point = ((high_surrogate - 0xD800) << 10) + (low_surrogate - 0xDC00) + 0x10000
            result += chr(code_point)
            i += 2
        else:
            result += string[i]
            i += 1
    return result


def fsw_init_package():
    if not os.path.exists('/sign_to_png/font_db'):
        os.makedirs('sign_to_png/font_db')

    with zipfile.ZipFile(r'../sign_writing_approach/font_db.zip', 'r') as zip_ref:
        zip_ref.extractall('sign_to_png')

    subprocess.call('npm install', cwd='sign_to_png/font_db', shell=True)


if __name__ == '__main__':
    import itertools

    fsw_init_package()
    with open(r'../sign_writing_approach/resources/sign2mint.jsonl', 'r') as f:
        sign2mint = [json.loads(s) for s in list(f)]
    pose_folder = 'Poses'
    video_folder = 'Videos'
    svg_folder = 'Svgs'

    if os.path.exists(pose_folder):
        shutil.rmtree(pose_folder)
    os.mkdir(pose_folder)

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.mkdir(video_folder)

    if os.path.exists(svg_folder):
        shutil.rmtree(svg_folder)
    os.mkdir(svg_folder)

    i = 18
    for datum in tqdm(itertools.islice(sign2mint, 277 * (i-1), 277 * i)):
        print('Generate swu')
        tr = [transcript for transcript in datum['captions'] if transcript['language'] == 'Sgnw'][0]['transcription']
        tr = decode_surrogates(tr)
        [transcript for transcript in datum['captions'] if transcript['language'] == 'Sgnw'][0]['transcription'] = tr

        print('Generate svg')
        uid = datum['doc']['uid']
        subprocess.call(f'node swu/swu-sign-svg {tr} ../../{svg_folder}/{uid}.svg',
                        cwd='sign_to_png/font_db', shell=True)

        print('Generate pose')
        video_name = datum['doc']['url'].split('.mp4')[0]
        pose_name = uid
        save_pose(video_name, f'Poses/{pose_name}')

        print('Generate video')
        pose_file_path = f'{pose_folder}/{pose_name}'
        video_file_path = f'{video_folder}/{pose_name}'
        save_video(pose_file_path, video_file_path)
