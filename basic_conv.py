import torch
from torch import nn
import torchvision
from torchvision import transforms

# load CIFAR-10 dataset with pytorch
# convert to tensor, normalize and flatten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Lambda(lambda x: torch.flatten(x)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_id = list(range(4000))
val_id = list(range(4000, 5000))
test_id = list(range(500))

# subset dataset and create dataloader with batch_size=1
train_sub_set = torch.utils.data.Subset(trainset, train_id)
val_sub_set = torch.utils.data.Subset(trainset, val_id)
test_sub_set = torch.utils.data.Subset(testset, test_id)

train_loader = torch.utils.data.DataLoader(train_sub_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sub_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_sub_set, batch_size=1, shuffle=True)

# check data size, should be CxHxW, class map only useful for visualization and sanity checks
image_size = trainset[0][0].size(0)
class_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
             9: 'truck'}


# Implement a fully connected model

# implement operations for our model
def activation(x):
    """
    Implement activation function with tanh()
    :param x: input tensor
    :return: output tensor equals element-wise tanh(x)
    """
    # tanh(x) function defined as
    act = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return act

def activation_grad(x):
    """
    Calculate the gradient of activation() respect to input x
    You need to find the maths representation of the derivative first
    :param x: input tensor
    :return: element-wise gradient of activation()
    """
    # derivative of tanh = 1 - (tanh(x))**2
    delta_act = 1 - (activation(x)) ** 2
    return delta_act

def cross_entropy(pred, label):
    """
    Calculate the cross entropy loss, L(pred, label)
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: the cross entropy loss, L(pred, label)
    """
    # get the softmax values of predicted tensor
    softmax = nn.Softmax(dim=1)
    soft_pred = softmax(pred)

    # calculate the entropy loss
    loss = 0
    for i in range(len(pred)):
        loss += torch.sum((-1 * (label[i]) * torch.log(soft_pred[i])), dim=0)
    return loss

def cross_entropy_grad(pred, label):
    """
    Calculate the gradient of cross entropy respect to pred
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: gradient of cross entropy respect to pred
    """
    # get the softmax values of predicted tensor
    softmax = nn.Softmax(dim=1)
    soft_pred = softmax(pred)

    delta_loss = soft_pred - label
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return delta_loss

def forward(w1, b1, w2, b2, x):
    """
    forward operation
    1. one linear layer followed by activation
    2. one linear layer followed by activation
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :param x: input tensor
    :return: x0, s1, x1, s2, x2
    """
    x0 = x

    s1 = torch.matmul(x0, torch.transpose(w1, 1, 0)) + b1
    x1 = activation(s1)
    s2 = torch.matmul(x1, torch.transpose(w2, 1, 0)) + b2
    x2 = activation(s2)

    return x0, s1, x1, s2, x2

def backward(w1, b1, w2, b2, t, x, s1, x1, s2, x2,
             grad_dw1, grad_db1, grad_dw2, grad_db2):
    """
    backward propagation, calculate dl_dw1, dl_db1, dl_dw2, dl_db2 using chain rule
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :param t: label
    :param x: input tensor
    :param s1:
    :param x1:
    :param s2:
    :param x2:
    :param grad_dw1: gradient of w1
    :param grad_db1: gradient of b1
    :param grad_dw2: gradient of w2
    :param grad_db2: gradient of b2
    :return:
    """
    x0 = x

    # Calculate grad_dx2 using x2, t
    grad_dx2 = x2 - t

    grad_ds2 = grad_dx2 * activation_grad(s2)

    grad_dx1 = torch.matmul(grad_ds2, w2)

    grad_ds1 = grad_dx1 * activation_grad(s1)

    grad_dw2.add_(torch.matmul(grad_ds2.t(), x1))

    grad_db2.add_(grad_ds2.sum(dim=0))

    grad_dw1.add_(torch.matmul(grad_ds1.t(), x0))

    grad_db1.add_(grad_ds1.sum(dim=0))
