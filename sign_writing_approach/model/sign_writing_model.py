import torch
from transformers import CLIPModel

from sign_writing_approach.clip_fine_tuning.clip_fine_tuning_lighting import CLIPTrainer


def contrastive_loss(image_rep, text_rep):
    bs = image_rep.size(0)
    device = image_rep.device
    similarity_matrix = torch.matmul(image_rep, text_rep.T)
    mask = torch.eye(bs, device=device).bool()
    sim_pos = torch.diagonal(similarity_matrix)
    sim_neg = similarity_matrix[~mask].view(bs, -1).max(dim=1).values
    loss = (-torch.log(sim_pos / (sim_pos + sim_neg))).mean()
    return loss

class SignWritingModel:
    def __init__(self, checkpoint_path: str):
        args = dict(
            model_name_or_path="openai/clip-vit-base-patch32",
            loss_fn=contrastive_loss
        )
        self.trainer: CLIPTrainer = CLIPTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,**args)
        self.similarity_model: CLIPModel = self.trainer.model
        self.processor = self.trainer.processor

    def sign_writing_signature(self, text: str, image) -> torch.Tensor:
        inputs = self.trainer.processor(text=text,images=image,return_tensors="pt")
        encoded_text = self.similarity_model.get_text_features(input_ids=inputs.input_ids)
        encoded_image = self.similarity_model.get_image_features(pixel_values=inputs.pixel_values)
        encoded_vector = torch.cat((encoded_image, encoded_text), dim=-1)
        return encoded_vector
