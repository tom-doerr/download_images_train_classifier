#!/usr/bin/env python3

'''
Trains a Pytorch classifier on the images in the images directory.
The class labels are the names of the subdirectories.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from sklearn.metrics import classification_report


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='The learning rate decay factor')
    parser.add_argument('--batch-size', type=int, default=64, help='The batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum factor')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='The weight decay factor')
    parser.add_argument('--batch-norm', action='store_true', help='Add batch normalization layers')
    parser.add_argument('--dropout', action='store_true', help='Add dropout layers')
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training')
    return parser.parse_args()


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print("output:", output)
        print("target:", target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 49 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), (correct / len(test_loader.dataset))))
    return test_loss, correct


def main():
    global args, device
    args = parse_arguments()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    print('Using device:', device)

    # Create model
    model = Net(args.batch_norm, args.dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Create a train_loader that loads the images in the images directory.

    # Make all images the same size
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.ImageFolder(root='./images', transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(root='./images', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test_loss, correct = test(model, test_loader)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), (correct / len(test_loader.dataset))))
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), 'model.pt')



class Net(nn.Module):
    def __init__(self, batch_norm, dropout):
        super(Net, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        else:
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        if self.dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Dropout(0)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.batch_norm:
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = self.dropout(x)
            x = x.view(-1, 128 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = self.dropout(x)
            x = x.view(-1, 128 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            x = self.fc4(x)
        return x


if __name__ == '__main__':
    main()
