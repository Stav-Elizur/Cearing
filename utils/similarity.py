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


def check_cosin(traget_vector: Tensor, data,chosen_paramerer='image_embedding') -> list:
    differences = []
    for dt in data:
        # dt = { "text", "image_text_encoded", "pose_url"}
        vec = np.array(dt[f'{chosen_paramerer}']).flatten()
        img = np.array(traget_vector.tolist()).flatten()
        cos_diff = diff(vec, img)
        differences.append((dt['word'], dt['uid'], cos_diff))

    differences = sorted(differences, key=lambda diff_images: diff_images[2],reverse=True)

    return differences
