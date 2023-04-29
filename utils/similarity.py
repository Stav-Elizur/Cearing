import json
import numpy as np

from torch import Tensor


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


def check_cosin(traget_vector: Tensor, file_encoding_name) -> list:
    with open(file_encoding_name) as f:
        data = list(f)
        data = [json.loads(s) for i, s in enumerate(data)]

    differences = []
    for dt in data:
        vec = np.array(dt['image_encoded']).flatten()
        img = np.array(traget_vector).flatten()
        cos_diff = diff(vec, img)
        differences.append((dt['id'], dt['text'], cos_diff))

    # for (image, text, diff_images) in sorted(differences, key=lambda diff_images: diff_images[2], reverse=True):
    #     print(f"Image id: {image} text:{text} Cos_Diff: {diff_images}")

    return differences
