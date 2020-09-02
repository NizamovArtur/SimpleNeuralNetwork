import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, key, normalize=False, title=f'Confusion  matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Normalized {key} confusion matrix ")
    else:
        print(f'Confusion {key} matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j]), fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data = np.genfromtxt('D:\\mnist-in-csv(1)\\mnist_train.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('D:\\mnist-in-csv(1)\\mnist_test.csv', delimiter=',', skip_header=1)
train_input = []
train_output = []
test_input = []
test_output = []
selection_input = []
selection_output = []
for i in range(len(data)):
    train_output.append(data[i][:1])
    train_input.append(data[i][1:])
train_input = np.array(train_input)
train_output = np.array(train_output)
for i in range(len(test_data)):
    test_output.append(test_data[i][:1])
    test_input.append(test_data[i][1:])
test_input = np.array(test_input) / np.amax(train_input)
test_output = np.array(test_output)
train_input = np.array(train_input) / np.amax(train_input)
train_output = np.array(train_output) / 10

train_output_3 = []
for j in range(len(train_output)):
    train_output_2 = []
    for i in range(10):
        k = 0
        if train_output[j] == i / 10:
            k = 1
        train_output_2.append(k)
    train_output_3.append(train_output_2)
train_output_3 = np.array(train_output_3)


# softmax activation function
def activation_softmax_func(X):
    e_x = np.exp(X - np.max(X))
    S = e_x / e_x.sum(axis=0)
    cache = X
    return S, cache


# RELU activation function
def activasion_relu_func(X):
    S = np.maximum(0, X)
    cache = X
    return S, cache


# derivative of the activation function softmax
def derivative_activation_func_softmax(dA, cache):
    Z = cache
    e_x = np.exp(Z - np.max(Z))
    s = e_x / e_x.sum(axis=0)
    dZ = dA * s * (1 - s)
    return dZ


# derivative of the activation function RELU
def derivative_activation_func_relu(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# Initializing random weights and zero bases
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# forward propagation
def feedforward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache


# activation function forward propagation
def activation_feedforward(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = feedforward(A_prev, W, b)
        A, activation_cache = activation_softmax_func(Z)

    elif activation == "relu":
        Z, linear_cache = feedforward(A_prev, W, b)
        A, activation_cache = activasion_relu_func(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


# Forward propagation of the entire neural network
def feedforward_with_H_hidden_layers(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = activation_feedforward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = activation_feedforward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
    return AL, caches


# Calculating the error
def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = AL.T
    Y = Y.T
    cost = np.sum(np.nan_to_num(-Y * np.log(AL) - (1 - Y) * np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost


# Backpropagation
def backrpopagation(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# Backpropagation activation function
def activation_backpropagation(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = derivative_activation_func_relu(dA, activation_cache)
        dA_prev, dW, db = backrpopagation(dZ, linear_cache)

    elif activation == "softmax":
        dZ = derivative_activation_func_softmax(dA, activation_cache)
        dA_prev, dW, db = backrpopagation(dZ, linear_cache)

    return dA_prev, dW, db


# Backpropagation for the entire neural network
def backpropagation_with_H_hidden_layers(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backpropagation(dAL,
                                                                                                      current_cache,
                                                                                                      activation="softmax")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backpropagation(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# Updating weights and bases
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def neural_network(X, Y, Z, C, hidden_layers, learning_rate, num_iterations, print_cost=False):
    np.random.seed(1)
    names = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    sum_3 = 0
    costs = []
    costs_test = []
    test_output_2 = {}
    parameters = initialize_parameters(hidden_layers)
    cmt = np.zeros([10, 10])
    cmt_2 = np.zeros([10, 10])
    for i in range(0, num_iterations):
        AL, caches = feedforward_with_H_hidden_layers(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backpropagation_with_H_hidden_layers(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10 == 0:
            print(f"Cost after iteration {i} {cost}")
        if print_cost and i % 10 == 0:
            costs.append(cost)
        if print_cost and i % 10 == 0:
            test_1, test_1_cache = feedforward_with_H_hidden_layers(Z, parameters)
            sum_1 = 0
            for j in range(len(test_1.T)):
                if max(test_1.T[j]) > 0.5:
                    if C[0][j] == list(test_1.T[j]).index(max(test_1.T[j])):
                        test_output_2[f'{j}'] = C[0][j]
                        continue
                    else:
                        sum_1 += 1
                        test_output_2[f'{j}'] = list(test_1.T[j]).index(max(test_1.T[j]))
                else:
                    sum_1 += 1
                    test_output_2[f'{j}'] = list(test_1.T[j]).index(max(test_1.T[j]))
            costs_test.append(100 - sum_1 / 100)

    for i in range(len(Z.T)):
        j, k = int(C[0][i]), int(test_output_2[f'{i}'])
        cmt[j, k] = cmt[j, k] + 1
        cmt[j, k] = int(cmt[j, k])
    for i in range(len(X.T)):
        j, k = int(list(Y.T[i]).index(max(Y.T[2]))), int(list(AL.T[i]).index(max(AL.T[i])))
        cmt_2[j, k] = cmt_2[j, k] + 1
        cmt_2[j, k] = int(cmt_2[j, k])
    for i in range(len(X.T)):
        j, k = int(list(Y.T[i]).index(max(Y.T[2]))), int(list(AL.T[i]).index(max(AL.T[i])))
        sum_3 += np.abs(j - k)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    print(f'Accuracy train = {100 - sum_3 / 600}')
    plt.show()
    print(f'Accuracy test = {100 - sum_1 / 100}')
    plt.plot(np.squeeze(costs_test))
    plt.ylabel('Accuracy')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    key = 'train'
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cmt_2, names, key, title=f'Confusion {key} matrix')  # print(cmt_2)
    key = 'test'
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cmt, names, key, title=f'Confusion {key} matrix')  # print(cmt)
    return parameters, cmt, cmt_2


X = train_input.T
Y = train_output_3.T
Z = test_input.T
C = test_output.T
hidden_layers = [784, 30, 30, 30, 30, 10]
parameters, cmt, cmt_2 = neural_network(X, Y, Z, C, hidden_layers=hidden_layers, learning_rate=0.333,
                                        num_iterations=160, print_cost=True)
