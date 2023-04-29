import pytorch_lightning as pl

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.optim.adamw import AdamW
import torch
from transformers.models.clip.modeling_clip import CLIPOutput

from clip_sw_dataset import ClipSWDataset, IMAGES_ZIP_NAME, BASE_SW_PATH
from sign_writing_approach.clip_fine_tuning.fine_tuning_clip import split_into_train_and_test
class CLIPTrainer(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate=3e-5,batch_size=64):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name_or_path)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def forward(self, input_ids, attention_mask, image):
        aaa:CLIPOutput = self.model(input_ids=input_ids, attention_mask=attention_mask, visual_inputs=image)
        a = aaa
        return
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch
        outputs = self(input_ids, attention_mask, image)
        logits_per_image, logits_per_text = outputs.logits.unbind(dim=1)
        loss = (torch.nn.functional.cross_entropy(logits_per_image, labels) +
                torch.nn.functional.cross_entropy(logits_per_text, labels)) / 2
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch
        outputs = self(input_ids, attention_mask, image)
        logits_per_image, logits_per_text = outputs.logits.unbind(dim=1)
        loss = (torch.nn.functional.cross_entropy(logits_per_image, labels) +
                torch.nn.functional.cross_entropy(logits_per_text, labels)) / 2
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def prepare_data(self):
      split_into_train_and_test("images")
      train_dataset = ClipSWDataset("images/train")
      test_dataset = ClipSWDataset("images/test")
      print(f"Len train: {len(train_dataset)}")
      print(f"Len test: {len(test_dataset)}")
      self.train_dataset = train_dataset
      self.test_dataset = test_dataset

        
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


c = CLIPTrainer('')
c.