import json
import zipfile
import os
import random
import shutil
import torch
import torch.nn as nn
import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary


from sign_writing.JsonDataset import JsonlDataset


# TODO: debug model and functions, change DIR of data loaders, and add KNN
def split_into_train_and_test():
    # user defined function to shuffle
    def shuffle_function():
        return 0.5

    # Set the path of the folder you want to split
    images_path = "images"

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    with zipfile.ZipFile(f'{images_path}.zip', 'r') as zip_ref:
        zip_ref.extractall(images_path)

    # Set the percentage of files you want in each folder
    train_percent = 90

    # Get a list of all the files in the folder
    all_files = os.listdir(images_path)

    # Calculate the number of files for each folder based on the percentage
    num_files_folder1 = int(len(all_files) * (train_percent / 100))

    # Shuffle the list of files randomly
    random.shuffle(all_files, shuffle_function)

    # Create the two folders to store the files
    train_path = os.path.join(images_path, "train")
    test_path = os.path.join(images_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Copy the files into the two folders based on the percentages
    for i, file_name in enumerate(tqdm.tqdm(all_files)):
        if i < num_files_folder1:
            shutil.copy2(os.path.join(images_path, file_name), train_path)
        else:
            shutil.copy2(os.path.join(images_path, file_name), test_path)
        os.remove(os.path.join(images_path, file_name))

def handle_pretrained_model():
    vgg_model = models.vgg19(pretrained=True)
    features = vgg_model.features

    for param in vgg_model.parameters():
        param.requires_grad = True
    feature_extractor = nn.Sequential(features,nn.MaxPool2d(7),nn.Flatten())

    return feature_extractor


def preprocessing(batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # train_dataset = torchvision.datasets.ImageFolder('train/', train_transforms)
    # test_dataset = torchvision.datasets.ImageFolder('test/', test_transforms)
    train_dataset = JsonlDataset("images/train")
    test_dataset = JsonlDataset("images/test")

    # loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def train(model, device, optimizer, scheduler, train_loader, test_loader, criterion=nn.CrossEntropyLoss(),
          num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            data_loader = train_loader
            model.train()
            if phase == 'test':
                data_loader = test_loader
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs.data == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader)*32
            epoch_acc = running_corrects.double() / len(data_loader)*32

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))

        print()


def transfer_learning():
    # split_into_train_and_test()
    vgg_model = handle_pretrained_model()


    criterion = nn.MSELoss()
    optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    batch_size=32
    num_epochs = 10

    train_loader,test_loader = preprocessing(batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = vgg_model.to(device)

    model = train(model=vgg_model,scheduler = exp_lr_scheduler,train_loader=train_loader,
                  test_loader=test_loader,device= device,
                optimizer=optimizer,criterion=criterion,
                num_epochs=num_epochs)


if __name__ == "__main__":
    transfer_learning()
