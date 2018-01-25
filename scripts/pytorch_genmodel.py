import torch
import torch.onnx
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing


## load mnist dataset
root = './data'
download = False
trans = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    def name(self):
        return 'lenet'

def run():
    model = LeNet().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0]))
        correct_cnt, ave_loss = 0, 0
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score = model(x)
            loss = criterion(score, target)
            _, pred_label = torch.max(score.data, 1)
            correct_cnt += (pred_label == target.data).sum()
            ave_loss += loss.data[0]
        accuracy = correct_cnt * 1.0 / len(test_loader) / batch_size
        ave_loss /= len(test_loader)
        print('==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy))
    dummy_input = Variable(torch.randn(1, 1, 28, 28)).cuda()
    # Invoke export 
    torch.onnx.export(model, dummy_input, "lenet.onnx", verbose=True)


if __name__ == '__main__':
   multiprocessing.freeze_support()
   run()
