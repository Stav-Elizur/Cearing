import os
import zipfile

import torch.utils.data as data
import torch
import json
from PIL import Image
import torchvision.transforms as transforms


class JsonlDataset(data.Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.image_encodings = []
        with zipfile.ZipFile('image_encodings.zip', 'r') as zip_ref:
            zip_ref.extractall('')
        with open('image_encodings.json') as f:
            image_encodings: list = json.loads(json.load(f))
            self.image_encodings = image_encodings

    def __len__(self):
        return len(self.image_encodings)

    def __getitem__(self, index):
        preprocess_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dict = self.image_encodings[index]
        # convert the data_dict to PyTorch tensors or numpy arrays
        # as needed for your model
        image_tensor = torch.tensor(preprocess_transforms(Image.open(os.path.join(self.filepath, data_dict["id"]))))
        label_tensor = torch.tensor(data_dict["label"])
        return image_tensor, label_tensor
