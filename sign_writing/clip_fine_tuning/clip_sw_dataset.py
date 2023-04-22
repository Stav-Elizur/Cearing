import os
import zipfile

import torch.utils.data as data
import json
from PIL import Image
import torchvision.transforms as transforms

from tqdm import tqdm

import torch.nn.functional as F


class ClipSWDataset(data.Dataset):
    def __init__(self, dir_path, processor, tokenizer):
        self.dir_path = dir_path
        self.image_info = []
        self.processor = processor
        self.tokenizer = tokenizer

        if not os.path.isfile('images_info.jsonl'):
            with zipfile.ZipFile('images_info.zip', 'r') as zip_ref:
                zip_ref.extractall('')

        with open('images_info.jsonl') as f:
            self.image_info = list(f)
            self.image_info = [json.loads(s) for s in self.image_info]

        self.image_info = list(
            filter(lambda image_info: os.path.isfile(os.path.join(dir_path, f"{image_info['uid']}.png"))
                                      and len(image_info['terms']) > 1
                                      and len(image_info['terms'][1].split()) == 1
                                      and image_info['assumed_spoken_language_code'] == 'en',
                   self.image_info))

        texts = [self.processor(text=img['terms'][1], return_tensors='pt')['input_ids'][0]
                 for img in self.image_info]
        self.padded_tensors = []

        max_len = max([t.shape[0] for t in texts])
        for t in tqdm(texts):
            padded_tensor = F.pad(t, (0, max_len - t.shape[0]), mode='constant', value=0)
            self.padded_tensors.append(padded_tensor)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        preprocess_transforms = transforms.Compose([
            transforms.Lambda(lambda rgba_img: rgba_img.convert('RGB')),
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        data_dict = self.image_info[index]

        curr_img = Image.open(os.path.join(self.dir_path, f"{data_dict['uid']}.png"))
        image_tensor = preprocess_transforms(curr_img)

        image_tensor = self.processor(images=image_tensor, return_tensors='pt')['pixel_values'][0]
        label_tensor = self.padded_tensors[index]

        return image_tensor, label_tensor
