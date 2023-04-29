import os

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from hamnosys_approach.dataset.data import load_dataset
from model.Iterative_decoder import IterativeGuidedPoseGenerationModel
from model.model_types import ConfigPoseEncoder, ConfigTextEncoder
from model.pose_encoder import PoseEncoderModel
from model.text_encoder import TextEncoderModel
from utils.data_tokenizers import HamNoSysTokenizer
from utils.train_utils import zero_pad_collator
import pytorch_lightning as pl

BATCH_SIZE = 64


def main():
    train_dataset = load_dataset(split="train[10:]")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)
    validation_dataset = load_dataset(split="train[:10]")
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    pose_encoder = PoseEncoderModel(ConfigPoseEncoder(pose_dims=(num_pose_joints, num_pose_dims), dropout=0))

    text_encoder = TextEncoderModel(ConfigTextEncoder(tokenizer=HamNoSysTokenizer()))

    # Model
    model = IterativeGuidedPoseGenerationModel(pose_encoder=pose_encoder,
                                               text_encoder=text_encoder,
                                               hidden_dim=128,
                                               learning_rate=1e-4,
                                               seq_len_loss_weight=2e-5,
                                               smoothness_loss_weight=1e-2,
                                               noise_epsilon=1e-3,
                                               num_steps=100)

    callbacks = []
    os.makedirs("models", exist_ok=True)

    callbacks.append(
        ModelCheckpoint(dirpath="models/" + '1',
                        filename="model",
                        verbose=True,
                        save_top_k=1,
                        monitor='train_loss',
                        mode='min'))

    trainer = pl.Trainer(max_epochs=5, callbacks=callbacks, accelerator='cpu', devices=1)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)


if __name__ == '__main__':
    main()
