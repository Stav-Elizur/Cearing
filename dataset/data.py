import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

def load_dataset():
    config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False, include_pose=None)
    dicta_sign = tfds.load(name='dicta_sign', builder_kwargs={"config": config})

    counter = 0
    for datum in dicta_sign["train"]:
        if datum['spoken_language'].numpy().decode('utf-8') == "BSL":
            counter += 1
            print(datum['hamnosys'].numpy(), datum['text'].numpy().decode('utf-8'))
    print(counter)
    pass

if __name__ == '__main__':
    load_dataset()