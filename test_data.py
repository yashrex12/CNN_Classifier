from basic_conv import *
from train_data import *

# Test accuracy
nb_test_errors = 0
for x_test, y_test in test_loader:

    forward(w1, b1, w2, b2, x_test)

    x_test_pred = x_test.max(dim=1)[1]

    nb_train_errors += torch.sum(x_test_pred != y_test).item()


# print test set error
print('test_accuracy {:.02f}%'.format(100 - (100 * nb_test_errors) / len(test_loader)))