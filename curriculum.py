import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., proportion=1):

        self.std = std
        self.mean = mean
        self.portion = proportion
        
    def __call__(self, tensor):
        if random.uniform(a=0, b=1) <= self.proportion or self.proportion == 1:
            tensor += torch.normal(mean=self.mean, std=self.std, size=tensor.size())
            tensor = torch.min(torch.ones(tensor.size()), tensor)
            tensor = torch.max(torch.zeros(tensor.size()), tensor)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

def train(criterion, model, loader, optimizer):
    for i, (images, labels) in enumerate(loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28)
        labels = labels
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def eval_loss_and_error(model, loader, device=None):
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, 28*28)
            labels = labels
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
    return acc
    

batch_size = 64
input_size = 784
hidden_sizes = [50, 100, 200, 400, 800, 1600]
drop_rate = 5
num_classes = 10
std = 0.2 ## standart deviation of a gaussian noise
learning_rate = 0.001
ratios = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5]
num_epochs = 500


torch.manual_seed(123)

GaussianNoise = AddGaussianNoise(mean=0, std=std)
PureTransform = transforms.Compose([transforms.ToTensor()])
GaussianTransform = transforms.Compose([transforms.ToTensor(), GaussianNoise])

pure_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=PureTransform, download=True)
perturbed_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=GaussianTransform, download=False)

pure_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=PureTransform, download=False)
perturbed_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=GaussianTransform, download=False)

size = 10000
indices = torch.randperm(size)
train_ind = indices

train_pure = Subset(pure_train_dataset, train_ind[:len(train_ind)//2])# splitting training data set into two parts: pure and perturbed
train_perturbed = Subset(perturbed_train_dataset, train_ind[len(train_ind)//2::])

mixed_train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([train_pure, train_perturbed]), batch_size=batch_size, shuffle = True)
 
pure_train_loader = torch.utils.data.DataLoader(dataset=train_pure, batch_size=batch_size, shuffle = True)
perturbed_train_loader = torch.utils.data.DataLoader(dataset=train_perturbed, batch_size=batch_size, shuffle = True)

pure_test_loader = torch.utils.data.DataLoader(dataset=pure_test_dataset, batch_size=batch_size, shuffle = True) 
perturbed_test_loader = torch.utils.data.DataLoader(dataset=perturbed_test_dataset, batch_size=batch_size, shuffle = True) 



total_n_data = len(mixed_train_loader)*num_epochs
for ratio_first_second_phase in ratios:
    num_epochs_first_phase = int((num_epochs * 2 * ratio_first_second_phase)/(ratio_first_second_phase + 1))
    num_epoch_second_phase = int((num_epochs * 2)/(ratio_first_second_phase + 1))
    f = open(f"results_curriculum_training{ratio_first_second_phase}n{num_epochs}.txt", "a")
    f.write(f"std: {std}, num_epochs{num_epochs}\n")

    for hidden_size in hidden_sizes:
        model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Training on pure dataset
        for epoch in range(num_epochs_first_phase):
            train(criterion=criterion, model=model, loader = pure_train_loader, optimizer=optimizer)
            if epoch%10 == 0:
                print(f"hidden size: {hidden_size}, {epoch/(num_epochs*2)*100}%")

        acc_pure = eval_loss_and_error(model=model,  loader=pure_test_loader)
        acc_perturbed = eval_loss_and_error(model=model, loader=perturbed_test_loader)
        f.write(f"After the first part of learning: Hidden size: {hidden_size}, accuracy on pure_set: {acc_pure}, accuracy on perturbed set: {acc_perturbed}\n")

        for g in optimizer.param_groups: # droping learning rate
                g['lr'] = learning_rate/drop_rate
        #Training on perturbed dataset
        for epoch in range(num_epoch_second_phase):
            train(criterion=criterion, model=model, loader=perturbed_train_loader, optimizer=optimizer)
            if epoch%10 == 0:
                print(f"hidden size: {hidden_size}, {(0.5+epoch/(num_epochs*2))*100}%")
        acc_pure = eval_loss_and_error(model=model,  loader=pure_test_loader)
        acc_perturbed = eval_loss_and_error(model=model, loader=perturbed_test_loader)
        f.write(f"Hidden size: {hidden_size}, accuracy on pure_set: {acc_pure}, accuracy on perturbed set: {acc_perturbed}\n")  
    f.close()




