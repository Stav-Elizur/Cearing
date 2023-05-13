import torch
from transformers import CLIPModel

from sign_writing_approach.clip_fine_tuning.clip_fine_tuning_lighting import CLIPTrainer


class SignWritingModel:
    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)

        # self.trainer: CLIPTrainer = CLIPTrainer.load_from_checkpoint(
        #     checkpoint_path=checkpoint_path)
        # self.similarity_model: CLIPModel = self.trainer.model
        # self.processor = self.trainer.processor

    def sign_writing_signature(self, text: str, image) -> torch.Tensor:
        encoded_text = self.model.encode_text(text=text)
        encoded_image = self.model.encode_image(image=image)
        encoded_vector = torch.cat((encoded_image, encoded_text), dim=-1)
        return encoded_vector
