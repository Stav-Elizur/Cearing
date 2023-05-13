import io
import json
import os
from zipfile import ZipFile

import cairosvg
from PIL import Image

from sign_writing_approach.model.sign_writing_model import SignWritingModel


def extract_sign2mint_zip(zip_path: str, dir_name: str):
    with ZipFile(zip_path, 'r') as zObject:
        zObject.extractall(path=dir_name)


def store_sign2mint_vectors(model: SignWritingModel, dir_images_path: str):
    with open('../resources/sign2mint.jsonl', 'r') as f:
        sign2mint = [json.loads(s) for s in list(f)]
        vectors_list = []
        for row in sign2mint:
            uid = row["doc"]["uid"]
            word = row["captions"][0]["transcription"]
            png_data = cairosvg.svg2png(url=f'{uid}.svg', write_to=None)
            curr_img = Image.open(io.BytesIO(png_data))

            embedding_vector = model.sign_writing_signature(
                text=word, image=curr_img)
            vectors_list.append(
                {"word": word, "uid": uid, "embedding_vector": embedding_vector})

        print(vectors_list)


if __name__ == '__main__':
    # extract_sign2mint_zip(zip_path='./sign2mint-Svgs.zip',dir_name='sign2mint-images')
    model = SignWritingModel('../model/sw_model.ckpt')
    store_sign2mint_vectors(model=model, dir_images_path="sign2mint-images")
