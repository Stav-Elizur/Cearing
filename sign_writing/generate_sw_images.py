import os
import zipfile
import json
import subprocess
from typing import List
import requests
from tqdm import tqdm


def fsw_init_package():
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
                    "n_best": 1,
                    "translation_type": "sent"
                })

                captions.append({"language": "Sgnw", "transcription": encoded_sw})

            transcription = captions[1]['transcription']

            subprocess.call(f'node fsw/fsw-sign-png  {transcription} ../../photos_results/{uid}.png',
                            cwd='sign_to_png/font_db', shell=True)
            json.dump(result, target_file)
            target_file.write('\n')


if __name__ == '__main__':
    fsw_init_package()
    generate_images_from_sw()
    clean_fsw_package()
