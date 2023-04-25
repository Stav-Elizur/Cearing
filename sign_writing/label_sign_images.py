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
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_info = []
    with open('images_info.jsonl') as f:
        image_info = [json.loads(s) for s in list(f)]
    with open('image_encodings.jsonl', 'w') as encoding_file:
        for image_file in tqdm(sorted(os.listdir(image_dir), key=lambda d: int(d.split('.')[0]))):
            # Load the image
            image = Image.open(os.path.join(image_dir, image_file))

            # Preprocess the image
            image = preprocess(image).unsqueeze(0).to(device)
            index = int(image_file.split('.')[0])

            if len(image_info[index]['terms']) > 1:

                # Load the text prompt
                text = image_info[index]['terms'][1]

                if len(text) < 77:
                    # Encode the text prompt
                    text = clip.tokenize(text).to(device)

                    # Perform the similarity check
                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        text_features = model.encode_text(text)
                        encoded_vector = torch.cat((image_features, text_features), dim=-1)
                        json_string = json.dumps({"id": image_file, "label": encoded_vector.tolist()})
                        encoding_file.write(json_string + '\n')


def check_cosin(filepath):
    with open("image_encodings.jsonl") as f:
        data = list(f)
        data = [json.loads(s) for s in data]

    image_features = [d['label'] for d in data if d['id'] == filepath]

    print(f"Diff with image: {filepath}")
    print("---------------------")
    differences = []
    # Todo: array then sort by diff {id: '*.png',diff:cos_diff}
    for dt in data:
        vec = np.array(dt['label']).flatten()
        img = np.array(image_features).flatten()
        cos_diff = diff(vec, img)

        differences.append((dt['id'], cos_diff))

    for (image, diff_images) in sorted(differences, key=lambda diff_images: diff_images[1], reverse=True):
        print(f"Image id: {image} Cos_Diff: {diff_images}")


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


if __name__ == '__main__':
    IMAGES_INFO_ZIP_NAME = "images_info.zip"
    IMAGES_ZIP_NAME = "images.zip"

    if not os.path.exists('images'):
        os.makedirs('images')

        with zipfile.ZipFile(IMAGES_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall('images')

    if not os.path.isfile('images_info.jsonl'):
        with zipfile.ZipFile(IMAGES_INFO_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall('')

    generate_labeling("images")
    # shutil.rmtree('images')
    # check_cosin(filepath="0.png")