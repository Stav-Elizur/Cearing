import os
import shutil
import zipfile
import random

from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.optim import AdamW
import torch
from transformers.models.clip.modeling_clip import CLIPOutput

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
        train_percent = 5

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

def preprocessing(batch_size=32,processor=None,tokenizer=None):
    split_into_train_and_test()
    train_dataset = ClipSWDataset("images\\train",processor,tokenizer=tokenizer)
    test_dataset = ClipSWDataset("images\\test",processor,tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    return train_loader, test_loader


def train(model: CLIPModel, clip_processor: CLIPProcessor, optimizer, dataloader,loss_fn):
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader):
        outputs:CLIPOutput = model(pixel_values=images,input_ids= labels)

        labels = torch.arange(outputs.logits_per_image.shape[0])
        image_loss = loss_fn(outputs.logits_per_image, labels)
        text_loss = loss_fn(outputs.logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == '__main__':
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    batch_size = 128
    num_epochs = 10
    train_loader, test_loader = preprocessing(batch_size=batch_size,processor=processor,tokenizer=tokenizer)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    for epoch in range(10):
        loss = train(model=model, clip_processor=processor, optimizer=optimizer, dataloader=train_loader,loss_fn=triplet_loss)
        print(f'Epoch {epoch}: loss={loss:.4f}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine-tuned-clip-model.pth')
