import torch
from torch.optim.lr_scheduler import StepLR
from model import CIFAR10_Model
from train import train
from validate import test
import torch.optim as optim
from torchsummary import summary
from dataloader import get_CIFAR10_data_loader
from checkpoint import save_model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = CIFAR10_Model().to(device)
summary(model, input_size=(3, 32, 32))


# train dataloader
train_loader, test_loader = get_CIFAR10_data_loader()

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

model =  CIFAR10_Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

EPOCHS = 100
for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    #scheduler.step()
    test(model, device, test_loader)

save_model(model, "CIFAR_10_model.pt")