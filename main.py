# Importing required libraries

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import os

# Run on GPU

device = torch.device("cuda:0")
print(device)
print('Using {} device'.format(device))


# Splitting test set to validation and test sets

class Dataset(Dataset):
    def __init__(self, text_file, root_dir, transform):
        self.name = pd.read_csv(text_file, sep=" ", usecols=range(1))
        self.label = pd.read_csv(text_file, sep=" ", usecols=range(1, 2))
        self.dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        image_name = os.path.join(self.dir, self.name.iloc[idx, 0])
        image = Image.open(f'./data/{image_name}')
        image = self.transform(image)
        labels = self.label.iloc[idx, 0]
        # labels = labels.reshape(-1, 2)
        return image, labels


transform_data = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
train_data = Dataset(text_file='data/splits/test.txt',
                     root_dir='./', transform=transform_data)
test_data = Dataset(text_file='data/splits/train.txt',
                    root_dir='./', transform=transform_data)
val_data = Dataset(text_file='data/splits/val.txt',
                   root_dir='./', transform=transform_data)
train_dataloader = DataLoader(train_data, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=True)
val_dataloader = DataLoader(val_data, shuffle=True)


# Mean and std of the dataset

def mean_and_std(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


# Normalize data

train_mean, train_std = mean_and_std(train_dataloader)
test_mean, test_std = mean_and_std(test_dataloader)
val_mean, val_std = mean_and_std(val_dataloader)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std),
])
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(val_mean, val_std),
])

train_data = Dataset(text_file='data/splits/test.txt',
                     root_dir='./', transform=train_transform)
test_data = Dataset(text_file='data/splits/train.txt',
                    root_dir='./', transform=test_transform)
val_data = Dataset(text_file='data/splits/val.txt',
                   root_dir='./', transform=val_transform)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)


# Building the Model

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training step

net = LeNet5()

# Hyperparameters

# loss_fn = nn.CrossEntropyLoss()
# lr = 1e-3
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# epochs = 10


# Training step

train_acc = []
train_losses = []


def train(epoch, model, loss_fn):
    # print('\nEpoch : %d' % epoch)

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for data in train_dataloader:
        inputs, labels = data[0], data[1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    accu = 100. * correct / total

    train_acc.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))


# Validation Step

val_losses = []
val_acc = []


def val(epoch, model, loss_fn):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data[0], data[1]

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(val_dataloader)
    accu = 100. * correct / total

    val_losses.append(test_loss)
    val_acc.append(accu)
    print('Val Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))


# Training and validating the model

# Hyperparameters

epochs = 20
loss_fn = nn.CrossEntropyLoss()
lr = 1e-3

for epoch in range(1, epochs + 1):
    if epoch > 0 and epoch % 20 == 0:
        lr = lr * 0.5  # Decaying Learning rate
    print('\nEpoch : %d' % epoch)
    print(f'Learning rate = {lr}')
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    train(epoch, net, loss_fn)
    val(epoch, net, loss_fn)

# Plotting Loss and Accuracy curves

plt.subplot(2, 1, 1)
plt.plot(train_losses, 'b')
plt.plot(val_losses, 'r')
plt.legend(["train", "val"])
plt.title('LOSS VALUES')
plt.subplot(2, 1, 2)
plt.plot(train_acc, 'b')
plt.plot(val_acc, 'r')
plt.legend(["train", "val"])
plt.title('Accuracies')
plt.show()


# Testing the model

def test_net(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # wrong = (predicted!= labels)

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return correct / total


test_net(net, test_dataloader)

# Confusion Matrix

n_classes = 10  # Number of  classes in the dataset

confusion_matrix = torch.zeros(n_classes, n_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_dataloader):
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
total_outputs = torch.sum(confusion_matrix, dim=1)
print('\n Confusion Matrix=:\n', confusion_matrix)

prob_matrix = torch.div(confusion_matrix, other=total_outputs)

print('\nProb_matrix = \n', prob_matrix)
a = 0
for i in range(len(prob_matrix[0])):
    a = prob_matrix[i][i] * 100
    print(f'Test accuracy of class {i} is: %.2f' % a, '%')  # Test accuracy per class
