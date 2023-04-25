import os
import json
import zipfile
import torch
import clip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

def generate_labeling(image_dir,image_encoding_path):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_info = []
    with open('images_info.jsonl') as f:
        image_info = [json.loads(s) for s in list(f)]
    with open(image_encoding_path, 'w') as encoding_file:
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

import matplotlib.pyplot as plt


def show_top_k(simliraty_arr, k,images_dir,result_path):
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=3)
    top_k = simliraty_arr[:k]
    photos = [plt.imread(os.path.join(images_dir, id)) for id,diff in top_k]
    # photos = [plt.imread(os.path.join(images_dir, id)) for id,text,diff in top_k]

    with open('images_info.jsonl') as f:
        image_info = [json.loads(s) for s in list(f)]

    # for i,(id,text,diff) in enumerate(top_k):
    for i,(id,diff) in enumerate(top_k):
        row, col = divmod(i, 3)
        axes[row, col].imshow(photos[i])
        text = image_info[int(id.split('.')[0])]["terms"][1]
        text = f'file name:{id}\ntext:{text}\ndiff:${diff}'
        # text = f'file name:{id}\ndiff:${diff}'
        print(text)
        axes[row, col].text(0.5, -0.3, f'{text}', transform=axes[row, col].transAxes, ha='center')

    # Remove the axis labels and ticks
    for ax in axes.ravel():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # Add a title for the entire figure
    fig.suptitle('Six Photos')

    # Save the plot to a file
    plt.savefig(result_path)

    # Display the plot
    plt.show()

def check_cosin(filepath,file_encoding_name):
    with open(file_encoding_name) as f:
        data = list(f)
        data = [json.loads(s) for i,s in enumerate(data) if i != 60 and i != 4101]

    image_features = [d['label'] for d in data if d['id'] == filepath]

    print(f"Diff with image: {filepath}")
    print("---------------------")
    differences = []
    for dt in data:
        vec = np.array(dt['label']).flatten()
        img = np.array(image_features).flatten()
        cos_diff = diff(vec, img)
        # differences.append((dt['id'],dt['text'], cos_diff))
        differences.append((dt['id'], cos_diff))

    # similarity = sorted(differences, key=lambda diff_images: diff_images[2], reverse=True)
    similarity = sorted(differences, key=lambda diff_images: diff_images[1], reverse=True)
    show_top_k(simliraty_arr=similarity,k=6,images_dir="images",result_path="subject.png")
    show_top_k(simliraty_arr=similarity[6:],k=6,images_dir="images",result_path='subject2.png')
    #
    # for (image, diff_images) in sorted(differences, key=lambda diff_images: diff_images[1], reverse=True):
    #     print(f"Image id: {image} Cos_Diff: {diff_images}")


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
    IMAGES_INFO_ZIP_NAME = "DataSets/Only English/images_info.zip"
    IMAGES_ZIP_NAME = "DataSets/Only English/images.zip"
    IMAGES_ENCODINGS_NAME = "DataSets/Only English/image_encodings.zip"

    if not os.path.exists('images'):
        os.makedirs('images')

        with zipfile.ZipFile(IMAGES_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall('images')

    if not os.path.isfile('images_info.jsonl'):
        with zipfile.ZipFile(IMAGES_INFO_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall('')

    if not os.path.isfile('image_encodings.jsonl'):
        with zipfile.ZipFile(IMAGES_ENCODINGS_NAME, 'r') as zip_ref:
            zip_ref.extractall('')

    # generate_labeling("images")
    # shutil.rmtree('images')

    check_cosin(filepath="4369.png",file_encoding_name="image_encodings.jsonl")
