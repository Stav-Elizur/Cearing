import os
import json
import shutil
import zipfile
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_labeling(image_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    sign_images_json = []
    for image_file in tqdm(os.listdir(image_dir)):
        image_processed = preprocess(Image.open(os.path.join(image_dir, image_file))).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_processed)
            sign_images_json.append({"id": image_file, "label": image_features.tolist()})
    with open('image_encodings.json', 'w') as f:
        json.dump(json.dumps(sign_images_json), f)


def diff_images(image_dir):
    with open(image_dir) as f:
        data = json.loads(json.load(f))
        a = np.array(data[0]['label']).flatten()
        print(data[0]['id'])
        print(data[1]['id'])
        print(data[2]['id'])

        b = np.array(data[1]['label']).flatten()
        c = np.array(data[2]['label']).flatten()
        d = np.array(data[3]['label']).flatten()
        diff(a, b)
        diff(a, c, True)
        diff(b, c)
        diff(a, d)
        diff(b, d)
        diff(c, d)


def check_cosin(images_dir, filepath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_processed = preprocess(Image.open(os.path.join(images_dir, filepath))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_processed).tolist()

    with open("image_encodings.json") as f:
        data = json.loads(json.load(f))
    max_image_diff = 0
    max_label = None
    # Todo: array then sort by diff {id: '*.png',diff:cos_diff}
    for dt in data:
        vec = np.array(dt['label']).flatten()
        img = np.array(image_features).flatten()
        cos_diff = diff(vec,img)
        if max_image_diff < cos_diff:
            max_image_diff = cos_diff
            max_label = dt
    print("Image id: ", max_label['id'], " Cos_Diff: ", max_image_diff)


def diff(a, b, similar=False):
    sim = ""
    if similar is not True:
        sim = "Not"
    dot_product = np.dot(a, b)

    # Calculate the magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)

    return cosine_similarity


if __name__ == "__main__":
    # if not os.path.exists('images'):
    #     os.makedirs('images')
    #
    # with zipfile.ZipFile('images.zip', 'r') as zip_ref:
    #     zip_ref.extractall('images')

    # generate_labeling("images")
    # shutil.rmtree('images')
    check_cosin(images_dir="images",filepath="0.png")
