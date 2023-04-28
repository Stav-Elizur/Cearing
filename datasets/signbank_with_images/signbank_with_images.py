"""SignBankWithImages dataset."""
import json
import os

import boto3

import tensorflow as tf
import tensorflow_datasets as tfds

from ..warning import dataset_warning
_CITATION = """
"""

_DESCRIPTION = """
SignBank Site: SignWriting Software for Sign Languages, including SignMaker 2017, 
SignPuddle Online, the SignWriting Character Viewer, SignWriting True Type Fonts, 
Delegs SignWriting Editor, SignBank Databases in FileMaker, SignWriting DocumentMaker, 
SignWriting Icon Server, the International SignWriting Alphabet (ISWA 2010) HTML Reference Guide, 
the ISWA 2010 Font Reference Library and the RAND Keyboard for SignWriting.
"""

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = 'signwriting-images'

client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)


class SignBankWithImages(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SignBank dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "puddle": tf.int32,
            "id": tfds.features.Text(),
            "assumed_spoken_language_code": tfds.features.Text(),
            "country_code": tfds.features.Text(),
            "created_date": tfds.features.Text(),
            "modified_date": tfds.features.Text(),
            "sign_writing": tfds.features.Sequence(tfds.features.Sequence(tfds.features.Text())),
            "sign_writing_images": tfds.features.Sequence(tfds.features.Sequence(tfds.features.Text())),
            "terms": tfds.features.Sequence(tfds.features.Text()),
            "user": tfds.features.Text(),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="http://signbank.org/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dataset_warning(self)

        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

        def __yielder():
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        yield obj['Key']

        return {
            "train": self._generate_examples(
                samples=__yielder()
            ),
        }

    def _generate_examples(self, samples):
        for i, sample in enumerate(samples):
            obj = client.get_object(Bucket=BUCKET_NAME, Key=sample)['Body'].read().decode('utf-8')
            obj = json.loads(obj)
            yield i, obj
