import os
import zipfile
import json
import subprocess
from typing import List
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

def generate_images_from_sign_bank():
    import sign_language_datasets.datasets
    import tensorflow_datasets as tfds
    import itertools
    import shutil

    if os.path.exists('photos_signbank_results'):
        shutil.rmtree('photos_signbank_results')

    os.mkdir('photos_signbank_results')

    signbank = tfds.load(name='sign_bank')
    signbank_train = list(filter(lambda datum: (len(datum['sign_writing'].numpy()) == 1) and
                                               (len(datum['sign_writing'].numpy()[0].decode('utf-8').split(' ')) == 1),
                                 tqdm(signbank['train'])))
    print(f'num of data: {len(signbank_train)}')

    num_of_files = 0
    for uid, datum in enumerate(tqdm(itertools.islice(signbank_train, 0, 10000))):
        sign_writing: List[bytes] = datum['sign_writing'].numpy()

        subprocess.call(f'node fsw/fsw-sign-png {sign_writing[0].decode("utf-8")} ../../photos_signbank_results/{uid}.png',
                        cwd='sign_to_png/font_db', shell=True)

        files = os.listdir('photos_signbank_results/')
        if len(files) != (num_of_files + 1):
            print("ERROR: Don't generate a file")
            exit(1)
        break

        num_of_files += 1


if __name__ == '__main__':
    fsw_init_package()
    generate_images_from_sign_bank()
    # generate_images_from_sw()
    clean_fsw_package()

    import shutil
    import os.path

    # Creating the ZIP file
    archived = shutil.make_archive('images', 'zip', 'photos_signbank_results')

    if os.path.exists('images.zip'):
        print(archived)
    else:
        print("ZIP file not created")
