import glob
import hashlib
import json
import os
import os.path
import uuid
import zipfile
import subprocess

from pathlib import Path

import boto3
import requests

import sign_language_datasets.datasets
import tensorflow_datasets as tfds
import itertools
import shutil

from tqdm import tqdm


def fsw_init_package():
    if not os.path.exists('sign_to_png/font_db'):
        os.makedirs('sign_to_png/font_db')

    with zipfile.ZipFile('font_db.zip', 'r') as zip_ref:
        zip_ref.extractall('sign_to_png')

    subprocess.call('npm install', cwd='sign_to_png/font_db', shell=True)


def api_call_spoken2sign(payload):
    url = 'https://pub.cl.uzh.ch/demo/signwriting/spoken2sign'
    response = requests.post(url, json=payload).json()
    return response['translations'][0]


ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = 'signwriting-images'


def get_md5(img_path: str):
    with open(img_path, 'rb') as f:
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

        return hash_md5.hexdigest()


def upload_image(client, img_path: str):
    file_md5 = get_md5(img_path)
    with open(img_path, 'rb') as f:

        object_key = f'signlanguage/{file_md5}.svg'

        client.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=f
        )

        return client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': object_key,
            }
        )


def datum_to_json(datum) -> dict:
    datum['id'] = datum['id'].numpy().decode('utf-8')
    datum['created_date'] = datum['created_date'].numpy().decode('utf-8')
    datum['modified_date'] = datum['modified_date'].numpy().decode('utf-8')
    datum['country_code'] = datum['country_code'].numpy().decode('utf-8')
    datum['assumed_spoken_language_code'] = datum['assumed_spoken_language_code'].numpy().decode('utf-8')
    datum['puddle'] = int(datum['puddle'].numpy())
    datum['sign_writing'] = [sign_writing.decode('utf-8') for sign_writing in datum['sign_writing'].numpy()]
    datum['user'] = datum['user'].numpy().decode('utf-8')
    datum['terms'] = [term.decode('utf-8') for term in datum['terms'].numpy()]
    return datum


def generate_images_from_sign_bank():
    client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    if os.path.exists('photos_signbank_results'):
        shutil.rmtree('photos_signbank_results')

    os.mkdir('photos_signbank_results')

    signbank = tfds.load(name='sign_bank')
    signbank_train = signbank['train']

    limit = 10_000
    with tqdm(range(limit)) as counter:
        for datum in itertools.islice(signbank_train, 0, limit):
            counter.update(1)

            sign_writings_lists: list[list[str]] = []
            sign_writing_images: list[list[str]] = []
            datum_id = datum['id'].numpy().decode('utf-8')

            if f'{datum_id}.json' in os.listdir(os.path.join(os.getcwd(), 'output')):
                continue

            if 'sign_writing' in datum:
                sign_writings_lists: list[list[str]] = [
                    sign_writing.decode('utf-8').split(' ')
                    for sign_writing in datum['sign_writing'].numpy()
                ]

            for sign_writings_list in sign_writings_lists:
                sign_writing_images_list: list[str] = []
                for sign_writing in sign_writings_list:
                    filename = uuid.uuid4().hex

                    command_result = subprocess.call(
                        f'node {os.path.join("./", "sign_to_png", "font_db", "fsw", "fsw-sign-svg")} '
                        f'{sign_writing} {os.path.join("./", "photos_signbank_results", f"{filename}.svg")}',
                        shell=True,
                    )

                    if command_result != 0:
                        continue

                    img_url = upload_image(
                        client=client,
                        img_path=os.path.abspath(f'./photos_signbank_results/{filename}.svg')
                    )

                    sign_writing_images_list.append(img_url)

                sign_writing_images.append(sign_writing_images_list)

            result = datum_to_json(datum)
            result['sign_writing'] = sign_writings_lists
            result['sign_writing_images'] = sign_writing_images

            with open(f'output/{result["id"]}.json', 'w') as f:
                json.dump(result, f)

            files = glob.glob('./photos_signbank_results/*')
            for f in files:
                os.remove(f)


if __name__ == '__main__':
    fsw_init_package()

    if not os.path.exists('output'):
        os.mkdir('output')

    generate_images_from_sign_bank()
