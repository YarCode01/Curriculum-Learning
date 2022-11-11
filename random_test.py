import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        tensor += torch.normal(mean=self.mean, std=self.std, size=tensor.size())
        tensor = torch.min(torch.ones(tensor.size()), tensor)
        tensor = torch.max(torch.zeros(tensor.size()), tensor)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoisePortion(object):
    def __init__(self, mean=0., std=1., portion=1):
        self.std = std
        self.mean = mean
        self.portion = portion
        
    def __call__(self, tensor):
        if 
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
num_classes = 10
std = 0.3 ## standart deviation of a gaussian noise
learning_rate = 0.001
num_epochs = 100
curriculum_learning = 1 # -1 for anticurriculum, 1 for curriculum, 0 for random, None for standart
arr_epochs_switch = [2,3]

torch.manual_seed(123)

GaussianNoise = AddGaussianNoise(mean=0, std=std)
PureTransform = transforms.Compose([transforms.ToTensor()])
GaussianTransform = transforms.Compose([transforms.ToTensor(), GaussianNoise])

pure_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=PureTransform, download=True)
perturbed_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=GaussianTransform, download=False)

pure_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=PureTransform, download=False)
perturbed_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=GaussianTransform, download=False)

size = len(pure_test_dataset)
indices = torch.randperm(size)
train_ind = indices[:int(0.8*size)] # 80 percent for the training set
test_ind = indices[int(0.8*size)::]

train_pure = Subset(pure_train_dataset, train_ind[:len(train_ind)//2])# splitting training data set into two parts: pure and perturbed
train_perturbed = Subset(perturbed_train_dataset, train_ind[len(train_ind)//2::])

test_pure = Subset(pure_train_dataset, train_ind)# using the same samples for testing, one test set is with noise, the other isn't. 
test_perturbed = Subset(perturbed_train_dataset, train_ind)

mixed_train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([train_pure, train_perturbed]), batch_size=batch_size, shuffle = True)

pure_train_loader = torch.utils.data.DataLoader(dataset=train_pure, batch_size=batch_size, shuffle = True)
perturbed_train_loader = torch.utils.data.DataLoader(dataset=train_perturbed, batch_size=batch_size, shuffle = True)
pure_test_loader = torch.utils.data.DataLoader(dataset=test_pure, batch_size=batch_size, shuffle = True) 
perturbed_test_loader = torch.utils.data.DataLoader(dataset=test_perturbed, batch_size=batch_size, shuffle = True) 



total_n_data = len(mixed_train_loader)*num_epochs
for num_epochs_switch in arr_epochs_switch:

    f = open(f"results_random_training_{num_epochs_switch}_seed.txt", "a")
    f.write(f"Num of epochs to switch:{num_epochs_switch}, total_epochs:{num_epochs}\n")
    f.write(f"std: {std}\n")

    for hidden_size in hidden_sizes:
        n_data = 0 
        model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        n_total_steps = len(mixed_train_loader)
        while n_data <= total_n_data:
            if curriculum_learning == 1:
                for i in range(num_epochs_switch):
                    train(criterion=criterion, model=model, loader = pure_train_loader, optimizer=optimizer)
                    n_data += len(pure_train_loader)
                curriculum_learning = -1
                
            elif curriculum_learning == -1:
                for i in range(num_epochs_switch):
                    train(criterion=criterion, model=model, loader = perturbed_train_loader, optimizer=optimizer)
                    n_data += len(perturbed_train_loader)
                curriculum_learning = 1
            print(f"{hidden_size}, {n_data/total_n_data * 100}%")

        acc_pure = eval_loss_and_error(model=model,  loader=pure_test_loader)
        acc_perturbed = eval_loss_and_error(model=model, loader=perturbed_test_loader)
        f.write(f"Hidden size: {hidden_size}, accuracy on pure_set: {acc_pure}, accuracy on perturbed set: {acc_perturbed}\n")

    f.close()


