#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from layers import Capsule
from utils import one_hot
from ipdb import set_trace


#
# Settings.
#

learning_rate = 0.001

batch_size = 128
test_batch_size = 128

# will display training loss and validation results after 10 steps
display_step = 10 

# choose whether validate during training steps, if not, validation will be executed after a full epoch
validate = True
# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

#
# Load MNIST dataset.
#

# Normalization for MNIST dataset.
dataset_transform = transforms.Compose([
                       transforms.ToTensor()
                       #transforms.Normalize((0.1307,), (0.3081,))
                       #transforms.Normalize((0.1787, 0.1737,0.1637), (0.031, 0.03, 0.029))
                   ])

#train_dataset = datasets.MNIST('./data', train=True, download=False, transform=dataset_transform)
train_dataset = datasets.ImageFolder('./train', transform = dataset_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#test_dataset = datasets.MNIST('./data', train=False, download=False, transform=dataset_transform)
test_dataset = datasets.ImageFolder('./test', transform = dataset_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

#
# Create capsule network.
#

image_height = 28
image_width = 28
image_channels = 3
conv_channels = 256
primary_channels = 32  # fixme get from conv2d
primary_capsule_num = 32 * 6 * 6
primary_vec_length = 8
digit_channels = 10 
digit_vec_length = 16
num_epochs = 100

def train(epoch):
    # add learning rate decay, since it converges quite fast, we decay the learning after every epoch
    optimizer = optim.Adam(network.parameters(), lr = learning_rate*(0.9**(epoch-1)))

    train_loss = None

    network.train()
    for ind, train_tup in enumerate(train_loader):
        data, label = train_tup
        label = one_hot(label, length=digit_channels)
        data= Variable(data).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        vec = network(data)

        loss = network.loss(vec, label, data)
        train_loss = loss.data[0]
        loss.backward()
        optimizer.step()

        #display loss and validation results after display_step steps, since validation is quite slow(will go through all 10000 test images), you can  
        if ind % display_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            ind * len(data),
            len(train_loader.dataset),
            100. * ind / len(train_loader),
            loss.data[0]))        
            #if validate: test(epoch)

    return train_loss

def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0

    for ind, test_tup in enumerate(test_loader):

        data, _label = test_tup
        label = one_hot(_label, length = digit_channels)
        data = Variable(data, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()

        vec = network(data)
        loss = network.loss(vec, label, data).cpu()
        test_loss = test_loss + loss.data[0]

        # v_mag -> [batch_size, digit_channels, 1, 1]
        v_mag = torch.sqrt((vec**2).sum(dim=2, keepdim=True))
        _, pred = torch.max(v_mag, 1)
        pred = pred.squeeze().cpu()
        
        correct += pred.data.eq(_label.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)*1.0 
    test_accuracy = 1.0*correct/len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * test_accuracy))

if __name__ =='__main__':

    network = Capsule(input_height=image_height,
                        input_width=image_width,
                        input_channels=image_channels,
                        conv_channels=conv_channels,
                        primary_vec_length=primary_vec_length,
                        primary_channels=primary_channels,
                        primary_capsule_num = primary_capsule_num,
                        digit_channels=digit_channels, 
                        digit_vec_length=digit_vec_length).cuda()

    for epoch in range(1, num_epochs + 1):
        train_loss = train(epoch)
        test(epoch)
        torch.save(network, './weights/model_'+str(epoch)+'.pt')
        if train_loss < early_stop_loss:
            break
        
