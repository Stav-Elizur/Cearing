import pytorch_lightning as pl

import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from transformers.models.clip.modeling_clip import CLIPOutput
from sign_writing_approach.clip_fine_tuning.clip_sw_dataset import ClipSWDataset

from sign_writing_approach.clip_fine_tuning.fine_tuning_clip import split_into_train_and_test


class CLIPTrainer(pl.LightningModule):
    def __init__(self, model_name_or_path, loss_fn, learning_rate=3e-5, batch_size=64, device='cuda'):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name_or_path)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.device = device

        self.model.vision_model.embeddings.position_embedding

    def forward(self, input_ids, image):
        return self.model(input_ids=input_ids, visual_inputs=image)

    def training_step(self, batch, batch_idx):
        image, texts = batch
        texts = [self.processor(text, return_tensors='pt').input_ids.to(
            self.device) for text in texts]

        # Pad tensors with zeros to make them the same size
        padded_tensors = []
        max_len = max([t.shape[1] for t in texts])
        for t in texts:
            padded_tensor = F.pad(
                t, (0, max_len - t.shape[1]), mode='constant', value=0)
            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.stack(padded_tensors)
        stacked_tensor = torch.squeeze(stacked_tensor, dim=1)

        outputs: CLIPOutput = self.model(stacked_tensor, image)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
        loss = self.loss_fn(logits_per_image, logits_per_text)
        self.log('train_loss', loss, on_epoch=True, logger=True,
                 prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        image, texts = batch
        texts = [self.processor(text, return_tensors='pt').input_ids.to(
            self.device) for text in texts]

        # Pad tensors with zeros to make them the same size
        padded_tensors = []
        max_len = max([t.shape[1] for t in texts])
        for t in texts:
            padded_tensor = F.pad(
                t, (0, max_len - t.shape[1]), mode='constant', value=0)
            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.stack(padded_tensors)
        stacked_tensor = torch.squeeze(stacked_tensor, dim=1)

        outputs: CLIPOutput = self.model(stacked_tensor, image)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
        loss = self.loss_fn(logits_per_image, logits_per_text)
        self.log('val_loss', loss, on_epoch=True, logger=True,
                 prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        train_dataset, test_dataset = split_into_train_and_test()
        print(f"Len Before train: {len(train_dataset)}")
        print(f"Len Before test: {len(test_dataset)}")
        self.train_dataset = ClipSWDataset(train_dataset)
        self.test_dataset = ClipSWDataset(test_dataset)
        print(f"Len After train: {len(self.train_dataset)}")
        print(f"Len After test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)
