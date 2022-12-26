import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def deriv_sigmoid(z):  
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return np.tanh(np.clip(z, -250, 250))

def deriv_tanh(z):
    return 1 - tanh(z) ** 2

def relu(z, epsilon=0.01):
    return np.maximum(0 + epsilon, z)

def deriv_relu(z, epsilon=0.01):
    Z = np.zeros_like(z)
    Z[z >= 0.0] = 1.0
    Z[z < 0.0] = 0.0 + epsilon
    return Z

def logistic_cost(y, y_pred):
    J = - (y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return J

def logistic_grad(y, y_pred, epsilon=1e-3):
    return -(np.divide(y, y_pred + epsilon) - np.divide(1.0 - y, 1.0 - y_pred + epsilon))

def mse(y, y_pred):
    MSE = (y - y_pred) ** 2
    return MSE

def mse_grad(y, y_pred):
    return (y_pred - y)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

def deriv_softmax(z):
    return softmax(z) * (1.0 - softmax(z))

def categorical_cross_entropy(y, y_pred):
    return -y * np.log(y_pred)

def grad_categorical_cross_entropy(y, y_pred):
    return (y_pred - y)

    