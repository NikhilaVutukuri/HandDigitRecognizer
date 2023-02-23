import sys
import numpy as np
import csv

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derviative for back propogation
def backward_sigmoid(x):
    return x * (1 - x)

#softmax activation for output layer
def softmax(x):
    y=np.exp(x)
    return y / np.sum(y, axis=0)

#loss function
def cross_entropy_loss(y, out):
    return -np.mean(np.multiply(y, np.log(out))+np.multiply(1-y+np.log(1-out)))

#forward propogation
def forward_pass(X):
    cache = dict()
    cache['Z1'] = np.dot(parameters['W1'], X)
    cache['A1'] = sigmoid(cache['Z1'])

    cache['Z2'] = np.dot(parameters['W2'], cache['A1'])
    cache['A2'] = sigmoid(cache['Z2'])

    cache['Z3'] = np.dot(parameters['W3'], cache['A2'])
    cache['A3'] = softmax(cache['Z3'])
    return cache


#training parameters
def train(X, Y):
    cache_forward = forward_pass(X)

    dA3 = Y - cache_forward['A3']
    dZ3 = dA3 * backward_sigmoid(cache_forward['A3'])
    dW3 = np.dot(dZ3, cache_forward['A2'].T)
    parameters['W3'] += learning_rate * dW3

    dA2 = np.dot(parameters['W3'].T, dA3)
    dZ2 = dA2 * backward_sigmoid(cache_forward['A2'])
    dW2 = np.dot(dZ2, cache_forward['A1'].T)
    parameters['W2'] += learning_rate * dW2

    dZ1 = np.dot(parameters['W2'].T, dA2) * backward_sigmoid(cache_forward['A1'])
    dW1 = np.dot(dZ1, X.T)
    parameters['W1'] += learning_rate * dW1

#preprocessing image matrix
def preprocess_images(file):
    with open(file) as img_file:
        return (np.array(list(csv.reader(img_file)), dtype=np.int32) / 255).astype('float32')

#preprocessing label matrix
def preprocess_labels(file):
    with open(file) as label:
        labels = np.array(list(csv.reader(label)), dtype=np.int32)
        oneHot = np.zeros((labels.shape[0], 10))
        for i in range(labels.shape[0]):
            oneHot[i][int(labels[i][0])] = 1

    return oneHot


if __name__ == '__main__':
    files = sys.argv

    #loading data
    train_images=preprocess_images(files[1])
    train_labels=preprocess_labels(files[2])
    test_images = preprocess_images(files[3])

    #hyperparameters
    learning_rate = 0.1
    epochs = 10

    #initializing parameters
    parameters =  {
        'W1': np.random.randn(256, 784) * 1 / np.sqrt(784),
        'W2': np.random.randn(64, 256) * 1 / np.sqrt(256),
        'W3': np.random.randn(10, 64) * 1 / np.sqrt(64)
        }

    for epoch in range(epochs):
        for i in range(len(train_images)):
            train(np.array(train_images[i], ndmin=2).T, np.array(train_labels[i], ndmin=2).T)

    preds = np.array([], dtype=np.int32)

    for i in range(len(test_images)):
        cache_after = forward_pass(test_images[i])
        output=cache_after['A3']
        result = np.squeeze(output.argmax(axis=0))
        preds = np.append(preds, int(result))

    #saving data to test_predictions.csv
    preds.tofile('test_predictions.csv', sep='\n')