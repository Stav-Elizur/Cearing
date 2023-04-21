import os
import shutil
import zipfile
import random
from torch import device
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer, AdamW
import torch

from clip_sw_dataset import ClipSWDataset


def split_into_train_and_test():
    # user defined function to shuffle
    def shuffle_function():
        return 0.5

    # Set the path of the folder you want to split
    images_path = "images"

    if not os.path.exists(images_path):
        os.makedirs(images_path)

        with zipfile.ZipFile(f'{images_path}.zip', 'r') as zip_ref:
            zip_ref.extractall(images_path)

        # Set the percentage of files you want in each folder
        train_percent = 90

        # Get a list of all the files in the folder
        all_files = os.listdir(images_path)

        # Calculate the number of files for each folder based on the percentage
        num_files_folder1 = int(len(all_files) * (train_percent / 100))

        # Shuffle the list of files randomly
        random.shuffle(all_files, shuffle_function)

        # Create the two folders to store the files
        train_path = os.path.join(images_path, "train")
        test_path = os.path.join(images_path, "test")
        os.makedirs(name=train_path,
                    exist_ok=True)
        os.makedirs(name=test_path,
                    exist_ok=True)

        # Copy the files into the two folders based on the percentages
        for i, file_name in enumerate(tqdm(all_files)):
            if i < num_files_folder1:
                shutil.copy2(os.path.join(images_path, file_name), train_path)
            else:
                shutil.copy2(os.path.join(images_path, file_name), test_path)
            os.remove(os.path.join(images_path, file_name))


def preprocessing(batch_size=32):
    split_into_train_and_test()
    train_dataset = ClipSWDataset("images\\train")
    test_dataset = ClipSWDataset("images\\test")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    return train_loader, test_loader


def train(model, clip_processor, optimizer, dataloader):
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader):
        images = [clip_processor(images=image, return_tensors='pt').to(device) for image in images]
        labels = clip_processor(labels, return_tensors='pt', padding=True).to(device)
        outputs: CLIPModel = model.encode_image_text(images, labels)
        loss = 1 - torch.sum(outputs) / outputs.shape[0]  # maximize similarity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


class ClipSWDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # Extract the images and labels from the batch
        images, labels = zip(*batch)

        # Convert the images to tensors and preprocess them
        images = [torch.tensor(img) for img in images]
        inputs = self.processor(images=images, text=list(labels), return_tensors='pt', padding=True)

        # Add pixel_values to the inputs dictionary
        inputs['pixel_values'] = torch.stack(images)

        return inputs

if __name__ == '__main__':
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    batch_size = 32
    num_epochs = 10
    train_loader, test_loader = preprocessing(batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # for images, labels in tqdm(train_loader):
    #     images = [processor(images=image, return_tensors='pt').to(device) for image in images]
    #     labels = processor(labels, return_tensors='pt', padding=True).to(device)
    #
    train_dataset = ClipSWDataset("images\\train")

    # # Define the fine-tuning parameters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
    )

    # Define the trainer and fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ClipSWDataCollator(processor),
    )
    trainer.train()

    # # Train the model
    # for epoch in range(10):
    #     loss = train(model=model, clip_processor=processor, optimizer=optimizer, dataloader=train_loader)
    #     print(f'Epoch {epoch}: loss={loss:.4f}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine-tuned-clip-model.pth')
