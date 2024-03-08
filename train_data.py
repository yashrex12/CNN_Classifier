from basic_conv import *

# training loop, we have 10 classes
nb_classes = 10
nb_train_samples = len(train_loader)

# set number of hidden neurons for first linear layer
nb_hidden = 50
# set learn rate and weights initialization std
lr = 1e-1 / nb_train_samples
init_std = 1e-6

# initialize weights and biases to small values from normal distribution
w1 = torch.empty(nb_hidden, image_size).normal_(0, init_std)
b1 = torch.empty(nb_hidden).normal_(0, init_std)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, init_std)
b2 = torch.empty(nb_classes).normal_(0, init_std)

# initialize empty tensor for gradients of weights and biases
grad_dw1 = torch.empty(w1.size())
grad_db1 = torch.empty(b1.size())
grad_dw2 = torch.empty(w2.size())
grad_db2 = torch.empty(b2.size())

# run for 1000 epochs
for k in range(1000):

    # initialize loss and train error counts
    acc_loss = 0
    nb_train_errors = 0

    # Clear all gradients of weights and biases
    grad_dw1 = grad_dw1.zero_()
    grad_db1 = grad_db1.zero_()
    grad_dw2 = grad_dw2.zero_()
    grad_db2 = grad_db2.zero_()

    # Accuracy on training data
    for x, y in train_loader:
        train_target_one_hot = nn.functional.one_hot(y.squeeze(dim=0), num_classes=nb_classes)

        # Forward propogation
        x0, s1, x1, s2, x2 = forward(w1, b1, w2, b2, x)

        # get the predictions
        x_pred = x2.max(dim=1)[1]

        # Accumulate training error and loss
        nb_train_errors += torch.sum(x_pred != y).item()
        acc_loss = cross_entropy(x2, train_target_one_hot).item()

        # backpropogation
        backward(w1, b1, w2, b2, train_target_one_hot, x, s1, x1, s2, x2,
                 grad_dw1, grad_db1, grad_dw2, grad_db2)

    # Step the gradient
    w1 -= lr * grad_dw1
    b1 -= lr * grad_db1
    w2 -= lr * grad_dw2
    b2 -= lr * grad_db2

    # Val error, initialize val error count
    nb_val_errors = 0

    # Accuracy on validation set
    for x_val, y_val in val_loader:
        forward(w1, b1, w2, b2, x_val)

        pred_x_val = x_val.max(dim=1)[1]

        nb_val_errors += torch.sum(pred_x_val != y_val).item()

        # print train and val information at end of each epoch
        print('{:d}: acc_train_loss {:.02f}, acc_train_accuracy {:.02f}%, val_accuracy {:.02f}%'
              .format(k,
                      acc_loss,
                      100 - (100 * nb_train_errors) / len(train_loader),
                      100 - (100 * nb_val_errors) / len(val_loader)))

