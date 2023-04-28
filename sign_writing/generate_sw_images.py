import glob
import os
import uuid
import zipfile
import json
import subprocess
from typing import List

import boto3
import requests
from tqdm import tqdm


def fsw_init_package():
    if not os.path.exists('sign_to_png/font_db'):
        os.makedirs('sign_to_png/font_db')

    with zipfile.ZipFile('font_db.zip', 'r') as zip_ref:
        zip_ref.extractall('sign_to_png')

    subprocess.call('npm install', cwd='sign_to_png/font_db', shell=True)


def clean_fsw_package():
    import shutil
    shutil.rmtree('sign_to_png')


def api_call_spoken2sign(payload):
    url = 'https://pub.cl.uzh.ch/demo/signwriting/spoken2sign'
    response = requests.post(url, json=payload).json()
    return response['translations'][0]


def generate_images_from_sw():
    if not os.path.exists('photos_results'):
        os.mkdir('photos_results')

    with open('signsuisse_source.jsonl', 'r') as json_file:
        json_list = list(json_file)

    num_of_files = 0
    with open('fixed_signsuisse.jsonl','w') as target_file:
        for json_str in tqdm(json_list):
            result = json.loads(json_str)

            captions: List[dict] = result['captions']
            uid = result['doc']['uid']

            if len(captions) != 2:
                transcription = captions[0]['transcription']
                country_code = captions[0]['language']
                language_code = result['doc']['meta']['language']
                encoded_sw = api_call_spoken2sign({
                    "country_code": country_code,
                    "language_code": language_code,
                    "text": transcription,
                    "translation_type": "sent"
                })

                captions.append({"language": "Sgnw", "transcription": encoded_sw})

            transcriptions = captions[1]['transcription']

            for i, transcription in enumerate(transcriptions.split(' ')):
                subprocess.call(f'node fsw/fsw-sign-png  {transcription} ../../photos_results/{uid}-{i}.png',
                                cwd='sign_to_png/font_db', shell=True)
                json.dump(result, target_file)

            files = os.listdir('photos_results/')
            if len(files) != (num_of_files + len(transcriptions.split(' '))):
                print("ERROR: Don't generate a file")
                exit(1)

            num_of_files += len(transcriptions.split(' '))
            target_file.write('\n')



ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')

BUCKET_NAME = 'signwriting-images'


def upload_image(client, img_path: str):
    from pathlib import Path

    img_name = Path(img_path).name
    object_key = f'signlanguage/{img_name}'

    with open(img_path, 'rb') as f:
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


def generate_images_from_sign_bank():
    import sign_language_datasets.datasets
    import tensorflow_datasets as tfds
    import itertools
    import shutil

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

    num_of_files = 0
    results = []

    for datum in itertools.islice(signbank_train, 0, 1000):
        sign_writing: List[bytes] = datum['sign_writing'].numpy()[0].decode('utf-8')
        filename = uuid.uuid4().hex

        subprocess.call(f'node fsw/fsw-sign-png {sign_writing} ../../photos_signbank_results/{filename}.png',
                        cwd='sign_to_png/font_db', shell=True)

        img_url = upload_image(client=client, img_path=os.path.abspath(f'./photos_signbank_results/{filename}.png'))

        files = os.listdir('photos_signbank_results/')
        if len(files) != (num_of_files + 1):
            print("ERROR: Don't generate a file")
            continue

        result = datum
        result['img_url'] = img_url
        results.append(result)

        files = glob.glob('./photos_signbank_results/*')
        for f in files:
            os.remove(f)

    print(results)


def create_semantic_images_html(word1,word2):
    import base64
    encoded_word1 = api_call_spoken2sign({
                    "country_code": 'us',
                    "language_code": 'us',
                    "text": word1,
                    "translation_type": "sent"
                })
    encoded_word2 = api_call_spoken2sign({
        "country_code": 'us',
        "language_code": 'us',
        "text": word2,
        "translation_type": "sent"
    })

    subprocess.call(f'node fsw/fsw-sign-png {encoded_word1} ../../photos_signbank_results/{word1}.png',
                    cwd='sign_to_png/font_db', shell=True)

    subprocess.call(f'node fsw/fsw-sign-png {encoded_word2} ../../photos_signbank_results/{word2}.png',
                    cwd='sign_to_png/font_db', shell=True)

    with open('output.html', 'w') as f:
        f.write('<html>\n<body>\n')

        with open(f'photos_signbank_results/{word1}.png', 'rb') as svg_file:
            svg_content = svg_file.read()
            encoded_svg = base64.b64encode(svg_content).decode('utf-8')
            f.write(f'<img src="photos_signbank_results/{word1}.png">\n')

        with open(f'photos_signbank_results/{word2}.png', 'rb') as svg_file:
            svg_content = svg_file.read()
            encoded_svg = base64.b64encode(svg_content).decode('utf-8')
            f.write(f'<img src="photos_signbank_results/{word2}.png">\n')
        f.write('</body>\n</html>\n')


if __name__ == '__main__':
    fsw_init_package()
    create_semantic_images_html('delicious','Tasty')
    # generate_images_from_sign_bank()
    # generate_images_from_sw()
    # clean_fsw_package()
