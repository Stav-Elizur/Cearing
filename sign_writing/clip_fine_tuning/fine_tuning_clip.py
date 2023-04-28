import os
import shutil
import zipfile
import random

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.optim.adamw import AdamW
import torch

from clip_sw_dataset import ClipSWDataset, IMAGES_ZIP_NAME, BASE_SW_PATH


def split_into_train_and_test(images_path):
    # user defined function to shuffle
    def shuffle_function():
        return 0.5

    # Set the path of the folder you want to split

    if not os.path.exists(images_path):
        os.makedirs(images_path)

        with zipfile.ZipFile(IMAGES_ZIP_NAME, 'r') as zip_ref:
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
    split_into_train_and_test(f"{BASE_SW_PATH}/images")
    train_dataset = ClipSWDataset(f"{BASE_SW_PATH}/images/train")
    test_dataset = ClipSWDataset(f"{BASE_SW_PATH}/images/test")

    print(f"Len train: {len(train_dataset)}")
    print(f"Len test: {len(test_dataset)}")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    return train_loader, test_loader


# Define the contrastive loss function
def contrastive_loss(image_rep, text_rep):
    bs = image_rep.size(0)
    device = image_rep.device
    similarity_matrix = torch.matmul(image_rep, text_rep.T)
    mask = torch.eye(bs, device=device).bool()
    sim_pos = torch.diagonal(similarity_matrix)
    sim_neg = similarity_matrix[~mask].view(bs, -1).max(dim=1).values
    loss = (-torch.log(sim_pos / (sim_pos + sim_neg))).mean()
    return loss

# Define the training loop
def train(model: CLIPModel, dataloader, optimizer:AdamW, contrastive_loss, processor, device):
    model.train()
    total_loss = 0.0
    for images, texts in tqdm(dataloader):
        images = images.to(device)
        texts = [processor(text, return_tensors='pt').input_ids.to(device) for text in texts]

        # Pad tensors with zeros to make them the same size
        padded_tensors = []
        max_len = max([t.shape[1] for t in texts])
        for t in texts:
            padded_tensor = F.pad(t, (0, max_len - t.shape[1]), mode='constant', value=0)
            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.stack(padded_tensors)
        stacked_tensor = torch.squeeze(stacked_tensor, dim=1)

        with torch.no_grad():
            text_rep = model.get_text_features(stacked_tensor)
        image_rep = model.get_image_features(images)

        loss: torch.Tensor = contrastive_loss(image_rep, text_rep)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


if __name__ == '__main__':
    device = torch.device('cpu')
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    batch_size = 64
    num_epochs = 10
    
    train_loader, test_loader = preprocessing(batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, train_loader, optimizer, contrastive_loss, processor, device)
        print(f'Epoch {epoch + 1} loss: {loss:.4f}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine-tuned-clip-model.pth')
    shutil.rmtree('images')
    os.remove("images_info.jsonl")