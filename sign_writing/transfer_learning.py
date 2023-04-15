import json
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

from sign_writing.JsonDataset import JsonlDataset


# TODO: debug model and functions, change DIR of data loaders, and add KNN

def handle_pretrained_model():
    vgg_model = models.vgg19(pretrained=True)
    features = vgg_model.features

    for param in vgg_model.parameters():
        param.requires_grad = False

    # num_features = vgg_model.classifier[-1].in_features
    # features = list(vgg_model.classifier.children())[:-1]  # Remove the last layer
    # features.extend([nn.Linear(num_features, 2)])  # Add our own custom layer
    # vgg_model.classifier = nn.Sequential(*features)
    feature_extractor = nn.Sequential(features, nn.Flatten())

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
    train_dataset = JsonlDataset("train")
    test_dataset = JsonlDataset("test")

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
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader)
            epoch_acc = running_corrects.double() / len(data_loader)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))

        print()


def transfer_learning():
    vgg_model = handle_pretrained_model()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(vgg_model.classifier.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #
    # batch_size=32
    # num_epochs = 10
    #
    # train_loader,test_loader = preprocessing(batch_size=batch_size)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg_model = vgg_model.to(device)
    #
    # model = train(model=vgg_model,scheduler = exp_lr_scheduler,train_loader=train_loader,
    #               test_loader=test_loader,device= device,
    #             optimizer=optimizer,criterion=criterion,
    #             num_epochs=num_epochs)


if __name__ == "__main__":
    transfer_learning()
