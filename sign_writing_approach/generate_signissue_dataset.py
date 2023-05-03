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


def fix_signissue_jsonl(with_sgnw: bool, with_images: bool):
    if with_images and not os.path.exists('photos_results'):
        os.mkdir('photos_results')

    with open(r'resources/signsuisse_source.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open(r'resources/fixed_signsuisse.jsonl','w') as target_file:
        for json_str in tqdm(json_list[0:15000]):
            result = json.loads(json_str)

            captions: List[dict] = result['captions']
            uid = result['doc']['uid']

            if with_sgnw and len(captions) != 2:
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

            if with_images:
                subprocess.call(f'node fsw/fsw-sign-png  {transcription} ../../photos_results/{uid}.png',
                                cwd='sign_to_png/font_db', shell=True)
            json.dump(result, target_file)
            target_file.write('\n')


if __name__ == '__main__':
    with_images = False
    with_sgnw = True

    if with_images:
        fsw_init_package()

    fix_signissue_jsonl(with_sgnw, with_images)

    if with_images:
        clean_fsw_package()
