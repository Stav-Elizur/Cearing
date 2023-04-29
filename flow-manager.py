from PIL import Image
from dataclasses import dataclass
import torch
from sign_writing_approach.clip_fine_tuning.clip_fine_tuning_lighting import CLIPTrainer
from sign_writing_approach.generate_sw_images import api_call_spoken2sign
from spoken_to_text.spoken_to_text import AudioRecorder
from transformers import CLIPModel
import subprocess
import cairosvg
import io
from utils.similarity import check_cosin


@dataclass
class ImageTextPair:
    image: Image
    text: str


class SignWritingModel:
    def __init__(self, checkpoint_path: str):
        self.trainer: CLIPTrainer = CLIPTrainer.load_from_checkpoint(
            checkpoint_path)
        self.similarity_model: CLIPModel = self.trainer.model
        self.processor = self.trainer.processor

    def sign_writing_signature(self, text: str, image) -> torch.Tensor:
        encoded_text = self.similarity_model.encode_text(text=text)
        encoded_image = self.similarity_model.encode_image(image=image)
        encoded_vector = torch.cat((encoded_image, encoded_text), dim=-1)
        return encoded_vector


class FlowManager:
    def __init__(self, model: SignWritingModel, audio_recorder: AudioRecorder):
        self.audio_recorder = audio_recorder
        self.model = model

    def generate_image_text_pairs(self, words_list: list[str]) -> list[ImageTextPair]:
        image_text_pairs = []
        for word in words_list:
            encoded_sw = api_call_spoken2sign({
                "country_code": 'us',
                "language_code": 'en',
                "text": word,
                "translation_type": "sent"
            })
            output = subprocess.check_output(
                f'node fsw/fsw-sign-png {encoded_sw}', cwd='sign_writing_approach/sign_to_png/font_db', shell=True)
            output_str = output.decode('utf-8')
            png_bytes = cairosvg.svg2png(output_str)
            curr_img = Image.open(io.BytesIO(png_bytes))
            image_text_pairs.append(ImageTextPair(
                **{'image': curr_img, 'text': word}))

        return image_text_pairs

    def get_similar_pose(similar_tensor_images):
        pass

    def run(self, audio_filename: str, encoded_vectors_path: str):
        spoken_text = self.audio_recorder.convert_to_text(
            filename=audio_filename)
        words_sentence: list[str] = spoken_text.split(' ')

        is_exist = False
        if is_exist is False:
            image_text_pairs: list[ImageTextPair] = self.generate_image_text_pairs(
                spoken_text=words_sentence)
            for image, text in image_text_pairs:
                encoded_vector: torch.Tensor = self.model.sign_writing_signature(
                    text=text, images=image)
                similar_tensor_images = check_cosin(traget_vector=encoded_vector,
                                                    file_encoding_name=encoded_vectors_path)
                pose = self.get_similar_pose(
                    similar_tensor_images=similar_tensor_images)


if __name__ == '__main__':
    recorder = AudioRecorder()
    flow_manager = FlowManager(
        model=SignWritingModel(''), audio_recorder=recorder)
    flow_manager.run(audio_filename='my_audio.wav', encoded_vectors_path='')
