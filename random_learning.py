import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., fraction=1):

        self.std = std
        self.mean = mean
        self.fraction = fraction
        
    def __call__(self, tensor):
        if random.uniform(a=0, b=1) <= self.fraction or self.fraction == 1:
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

def train(criterion, model, loader, optimizer, device=None):
    for i, (images, labels) in enumerate(loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

# def eval_loss_and_error(loss, model, loader, device=None):
#     n_correct = 0
#     n_samples = 0
#     l = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.reshape(-1, 28*28)
#             output = model(images)
#             l += loss(output, labels, reduction='sum').item()
#             _, predicted = torch.max(output.data, 1)
#             n_samples += labels.size(0)
#             n_correct += (predicted == labels).sum().item()

#         acc = 100.0 * n_correct / n_samples
#     return acc

def eval_loss_and_error(criterion, model, loader, device=None):
    l, accuracy, ndata = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data = data.reshape(-1, 28*28)
            data, target = data.to(device), target.to(device)
            output = model(data)
            l += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            ndata += len(data)
    
    return l/ndata, (1-accuracy/ndata)*100

def report(epoch, optimizer, criterion, model, train_loader, pure_test_loader, perturbed_test_loader, device):
    o = dict() # store observations
    o["epoch"] = epoch
    o["lr"] = optimizer.param_groups[0]["lr"]
    o["train_loss"], o["train_error"] = \
        eval_loss_and_error(criterion=criterion, model=model, loader=train_loader, device=device)
    o["test_loss_pure"], o["test_error_pure"] = \
        eval_loss_and_error(criterion=criterion, model=model, loader=pure_test_loader, device=device)
    o["test_loss_pertubed"], o["test_error_pertubed"] = \
        eval_loss_and_error(criterion=criterion, model=model, loader=perturbed_test_loader, device=device)

    for k in o:
        writer.add_scalar(k, o[k], epoch)
    

batch_size = 64
input_size = 784
hidden_sizes = [15, 30, 60, 100, 200]
drop_rate = 10
num_classes = 10
std = 0.2 ## standart deviation of a gaussian noise
learning_rate = 0.001
num_epochs = 300
size = 10000
fraction = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda" if use_cuda else "cpu")

print(f"USE_CUDA = {use_cuda},  DEVICE_COUNT={torch.cuda.device_count()}, NUM_CPU_THREADS={torch.get_num_threads()}")

torch.manual_seed(123)

GaussianNoise_half = AddGaussianNoise(mean=0, std=std, fraction = fraction)
PureTransform = transforms.Compose([transforms.ToTensor()])
GaussianTransform_half = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean=0, std=std, fraction=fraction)])
GaussianTransform_full = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean=0, std=std, fraction=1)])

perturbed_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean=0, std=std, fraction=fraction)]), download=False)

pure_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=PureTransform, download=False)
perturbed_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=GaussianTransform_full, download=False)


indices = torch.randperm(size)
train_ind = indices

train_mixed_first = Subset(perturbed_train_dataset, train_ind[:len(train_ind)//2])# splitting training data set into two parts: pure and perturbed
train_mixed_second = Subset(perturbed_train_dataset, train_ind[len(train_ind)//2::])

first_loader = torch.utils.data.DataLoader(dataset=train_mixed_first, batch_size=batch_size, shuffle = True)
second_loader = torch.utils.data.DataLoader(dataset=train_mixed_second, batch_size=batch_size, shuffle = True)

pure_test_loader = torch.utils.data.DataLoader(dataset=pure_test_dataset, batch_size=batch_size, shuffle = True) 
perturbed_test_loader = torch.utils.data.DataLoader(dataset=perturbed_test_dataset, batch_size=batch_size, shuffle = True) 

for hidden_size in hidden_sizes:
    writer = SummaryWriter(log_dir=f"results/Random, hidden_size={hidden_size}, learning_rate={learning_rate},num_epochs={num_epochs}, train_size={size}")
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Training on the first half
    for epoch in range(num_epochs):
        train(criterion=criterion, model=model, loader = first_loader, optimizer=optimizer, device=device)
        if epoch%10 == 0:
            print(f"{epoch/num_epochs/2*100}")
            report(epoch = epoch//2, optimizer = optimizer, criterion = criterion, model = model, train_loader = first_loader, pure_test_loader = pure_test_loader, perturbed_test_loader = perturbed_test_loader, device=device)
       
    for g in optimizer.param_groups: # droping learning rate
        g['lr'] = learning_rate/drop_rate

    #Training on the second half
    for epoch in range(num_epochs):
        train(criterion=criterion, model=model, loader=second_loader, optimizer=optimizer, device=device)
        if epoch%10 == 0:
            print(f"{(0.5 + epoch/(num_epochs*2))*100}")
            report(epoch = num_epochs//2 + epoch//2, optimizer=optimizer, criterion=criterion, model=model, train_loader=second_loader, pure_test_loader= pure_test_loader, perturbed_test_loader = perturbed_test_loader, device = device)





