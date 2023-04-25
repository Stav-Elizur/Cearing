import tensorflow_datasets as tfds
from datasets.signbank_with_images import SignBankWithImages

signbank = tfds.load(name='sign_bank_with_images')
signbank_train = signbank['train']


for datum in signbank_train:
    print(datum)
