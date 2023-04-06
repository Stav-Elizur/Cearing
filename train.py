from torch.utils.data import DataLoader

from dataset.data import load_dataset
from utils.train_utils import zero_pad_collator

BATCH_SIZE = 64


def main():
    train_dataset = load_dataset(split="train[10:]")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=zero_pad_collator)
    validation_dataset = load_dataset(split="train[:10]")
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn=zero_pad_collator)

    pass


if __name__ == '__main__':
    main()

