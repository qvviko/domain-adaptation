import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose, Normalize

# Make output deterministic
import numpy as np
np.random.seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(100)

batch_size = 32
test_batch_size = 100

# Transformations for mnist - transform to tensor and normalize
mnist_transformations = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# Transformations for svhn - resize to 28x28 and grayscale (as in mnist) - then normalize
svhn_transformations = Compose([
    Resize(28),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# Data Sources
svhn_train = datasets.SVHN('../data', split='train', download=True,
                           transform=svhn_transformations)
svhn_test = datasets.SVHN('../data', split='test', download=True,
                          transform=svhn_transformations)
mnist_test = datasets.MNIST('../data', train=False, download=True,
                            transform=mnist_transformations)

# Data loaders
svhn_train_loader = DataLoader(svhn_train,
                               batch_size=batch_size, shuffle=True)
svhn_test_loader = DataLoader(svhn_test,
                              batch_size=test_batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test,
                               batch_size=test_batch_size, shuffle=True)


# Simple CNN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Sequence with convolutional layers
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        # Sequence with linear layers
        self.seq2 = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.seq1(x)
        x = x.view(-1, 1600)
        return self.seq2(x)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_cnn = Net().to(device)


# Xavier initialization for the network
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

# Apply the init recursively
model_cnn.apply(init_weights)

# Function for training the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# Function for testing our model, returns the accuracy in %
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return 100 * correct / len(test_loader.dataset)


epochs = 10
lr = 0.0001
model = model_cnn
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

for epoch in range(1, epochs + 1):
    train(model, device, svhn_train_loader, optimizer, epoch)
    torch.save(model.state_dict(), "mnist_inno.pt")

print(test(model, device, mnist_test_loader))
