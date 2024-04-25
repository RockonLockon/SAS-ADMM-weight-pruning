import torch.nn.init
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    cnn = ResNet(BasicBlock, [5, 5, 5])
    cnn.dictKeys = []
    for i in cnn.state_dict().keys():
        if 'conv' in i:
            cnn.dictKeys.append(i)
    cnn.dictKeys.append('linear.weight')
    return cnn


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    cnn = ResNet(BasicBlock, [9, 9, 9])
    cnn.dictKeys = []
    for i in cnn.state_dict().keys():
        if 'conv' in i:
            cnn.dictKeys.append(i)
    cnn.dictKeys.append('linear.weight')
    return cnn

def assign_W(cnn,W_matrix):
    D = dict(cnn.named_parameters())
    for i,w in enumerate(W_matrix):
        D[cnn.dictKeys[i]].data = w


def get_W_matrix(cnn, zeroFlag=False):
    if zeroFlag == False:
        W_matrix = [cnn.state_dict()[i] for i in cnn.dictKeys]
    else:
        W_matrix = [torch.zeros_like(cnn.state_dict()[i]) for i in cnn.dictKeys]
    return W_matrix


def get_W_matrix_grad(cnn):
    D = dict(cnn.named_parameters())
    W_matrix_grad = [D[i].grad for i in cnn.dictKeys]
    return W_matrix_grad


def W_grad_keep_zero(cnn, Wzero_index):
    D = dict(cnn.named_parameters())
    for i,w in enumerate(Wzero_index):
        D[cnn.dictKeys[i]].grad *= w

def dataSeting(dataPath,batch_size, workers=0, pin_mem=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_train = datasets.CIFAR10(root=dataPath, train=True, download=False,transform=transform_train)
    cifar10_test = datasets.CIFAR10(root=dataPath, train=False, download=False,transform=transform_test)

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=workers,pin_memory=pin_mem)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=workers,pin_memory=pin_mem)
    return cifar10_train, cifar10_test, train_loader, test_loader

def dataSeting2(dataPath, batch_size, workers=0, pin_mem=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_train = datasets.CIFAR10(root=dataPath, train=True, download=True,transform=transform_train)
    cifar10_test = datasets.CIFAR10(root=dataPath, train=False, download=True,transform=transform_test)

    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_mem)
    return cifar10_train, cifar10_test, test_loader


def plot_hist(cnn,saveName = False):
    W_matrix = get_W_matrix(cnn)
    count = 0
    plt.figure(figsize=(3,24))
    for i in W_matrix:
        i = i.cpu()
        tmpW = i.view(-1).numpy()
        ax = plt.subplot(len(W_matrix), 1, count+1)
        ax.set_ylabel(cnn.dictKeys[count])

        plt.hist(tmpW, bins=1000, density=True)
        count += 1
    ax.set_xlabel("Weight")
    if saveName == False:
        plt.show()
    else:
        plt.savefig(saveName)
        plt.clf()
