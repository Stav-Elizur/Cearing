import os
import torch.utils.data as data
import torch
import json
from PIL import Image
import torchvision.transforms as transforms


class ClipSWDataset(data.Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.image_info = []
        with open('images_info.json') as f:
            self.image_info: list = json.load(f)
        self.image_info = list(filter(lambda image_info: os.path.isfile(os.path.join(dir_path, f"{image_info['uid']}.png")) and len(image_info['terms']) > 1,
                                      self.image_info))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        preprocess_transforms = transforms.Compose([
            transforms.Lambda(lambda rgba_img: rgba_img.convert('RGB')),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data_dict = self.image_info[index]

        curr_img = Image.open(os.path.join(self.dir_path, f"{data_dict['uid']}.png"))
        image_tensor = preprocess_transforms(curr_img)
        label_tensor = data_dict["terms"][1]
        if not isinstance(label_tensor,str):
            print('ss')
        return image_tensor, label_tensor
