import json
import os
import shutil

from PIL import Image
from dataclasses import dataclass
import torch
from sign_writing_approach.clip_fine_tuning.clip_fine_tuning_lighting import CLIPTrainer
from sign_writing_approach.generate_sw_images import api_call_spoken2sign
from sign_writing_approach.model.sign_writing_model import SignWritingModel
from spoken_to_text.spoken_to_text import AudioRecorder
from transformers import CLIPModel
import subprocess
import cairosvg
import io

from utils.pose_utils import save_pose_as_video, concate_two_videos
from utils.similarity import check_cosin


class FlowManager:
    def __init__(self, model: SignWritingModel,
                 encoded_vectors_path: str):
        self.model = model

        with open(encoded_vectors_path,'r') as f:
            self.data = [json.loads(s) for s in list(f)]

    def generate_image(self, spoken_word: str) -> Image:
        encoded_sw = api_call_spoken2sign({
            "country_code": 'us',
            "language_code": 'en',
            "text": spoken_word,
            "translation_type": "sent"
        })
        output = subprocess.check_output(
            f'node fsw/fsw-sign-svg {encoded_sw}', cwd='sign_writing_approach/sign_to_png/font_db', shell=True)
        output_str = output.decode('utf-8')
        png_bytes = cairosvg.svg2png(output_str)
        curr_img = Image.open(io.BytesIO(png_bytes))

        return curr_img

    def get_word_pose_url(self, word: str, data):
        for dt in data:
            if dt['word'] == word:
                return f'https://s3-eu-central-1.amazonaws.com/sign2mint/poses/{dt["uid"]}.pose'

        return None

    def run(self, sentence: str):
        words_sentence: list[str] = sentence.replace('`', '').split(' ')

        for text in words_sentence:
            pose_url = self.get_word_pose_url(text, self.data)
            if pose_url:
                save_pose_as_video(pose_url=pose_url, video_name=text)
            else:
                image: Image = self.generate_image(spoken_word=text)
                encoded_vector: torch.Tensor = self.model.sign_writing_signature(
                    text=text, image=image)
                similar_tensor_images = check_cosin(traget_vector=encoded_vector,
                                                    data=self.data)
                save_pose_as_video(
                    pose_url=similar_tensor_images[0]['uid'], video_name=text)

        final_video_name = 'final_video.mp4'
        video_files_name: list = os.listdir('videos')
        if len(video_files_name) > 0:
            os.rename(video_files_name[0], final_video_name)
        for video_file_name in video_files_name[1:]:
            concate_two_videos(
                final_video_name, video_file_name, final_video_name)

        shutil.rmtree('videos')


if __name__ == '__main__':
    audio_file_name = 'my_audio.wav'
    checkpoint_path = './sign_writing_approach/model/sw_model.ckpt'
    # recorder = AudioRecorder()
    flow_manager = FlowManager(model=SignWritingModel(
        checkpoint_path=checkpoint_path), encoded_vectors_path='./sign_writing_approach/store_vectors/sign2mint-vectors.jsonl')
    flow_manager.run('aaa')
