# logistic regression implementation with numpy

import numpy as np

# training data and labels
X_train = [[1,1,1],[1,2,2],[1,13,13],[1,14,14]]
Y_train = [[1,1,0,0]]
X_test = [[1,1,2]]

# training data and labels init
X = np.array(X_train, dtype=np.float32)
Y = np.array(Y_train, dtype=np.float32)
Y = np.transpose(Y)

def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))

# forward pass
def forward_pass(x, w):
    return logistic_sigmoid(np.matmul(x, w))

# gradient computation
def backward_pass(x, y, y_real):
    # chain rule
    return np.matmul(np.transpose(x), y - y_real)

# computing loss
def loss(y, y_real):
    # CE
    return (np.log(y) * y_real + (1 - y_real) * np.log(1 - y)).sum() * (-1)

# training
def train():
    w = np.array([[0.0],[0.0],[0.0]], dtype=np.float32)
    learning_rate = 0.1
    i = 50
    test_number = np.array(X_test, dtype=np.float32)

    for epoch in range(i):
        y = forward_pass(X, w)
        gradient = backward_pass(X, y, Y)
        w = w - learning_rate * gradient

        print(f'epoch {epoch + 1}, x = {test_number}, y = {forward_pass(test_number, w)[0][0]:.3f}, loss = {loss(y, Y):.3f}')

train()
