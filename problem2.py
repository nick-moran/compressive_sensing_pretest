import numpy as np
import random
import pickle
import gzip
import matplotlib.pyplot as plt
from copy import deepcopy

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y
                                    in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a;

    def update_mini_batch(self, mini_batch, eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            db, dW = self.backprop(x,y)
            grad_b = [gI + dI for gI, dI in zip(grad_b, db)]
            grad_w = [gI + wI for gI, wI in zip(grad_w, dW)]
        
        self.weights = [w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                            for b, nb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1],y) * \
                    sigmoid_prime(zs[-1])
        
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        #this updates backwards, taking into account that
        #indices can be negative in python.
        #ex, l=1 is the last layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta;
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (grad_b, grad_w)

    def cost_derivative(self, output, y):
        return output-y

    def evaluate(self, test_data):
        test_results = [np.linalg.norm(y-self.feedforward(x),ord=2)
                            for (x,y) in test_data]
        return sum(test_results)/len(test_data)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
                test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)

        for j in range(epochs):
            print(f'Epoch: {j}')
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # if test_data:
            #     print(f"Epoch {j} Loss: {self.evaluate(test_data)}")
            # else:
            #     print(f"Epoch {j} complete")
            if j % 10 == 0:
                print(f"Epoch {j} L2 norm: {self.evaluate(test_data)}")

        print(f'Final l2 norm: {self.evaluate(test_data)}')
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def load_data(file):
    f = gzip.open(file, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def data_handler(file,m):
    tr_d, va_d, te_d = load_data(file)
    #creating the sensing matrix...
    n = tr_d[0][0].shape[0]
    sensing_matrix = np.random.normal(0.0, 1.0/m, m*n)
    sensing_matrix = np.reshape(sensing_matrix, (m,n))

    data_rounded = [np.round_(img) for img in tr_d[0]]
    data_rounded = [np.reshape(x, (784,1)) for x in data_rounded]
    training_inputs = [sensing_matrix.dot(x) for x in data_rounded]
    training_data = zip(training_inputs, data_rounded)


    test_data_rounded = [np.round_(img) for img in te_d[0]]
    test_data_rounded = [np.reshape(x, (784,1)) for x in test_data_rounded]
    test_training_inputs = [sensing_matrix.dot(x) for x in test_data_rounded]
    test_data = zip(test_training_inputs, test_data_rounded)

    return (training_data, test_data)

#Location of the dataset...
training_data, test_data = data_handler('./mnist.pkl.gz',49)

net = Network([49,100,150,784])

lTemp = deepcopy(test_data)
lTemp1 = deepcopy(test_data)
lTemp2 = deepcopy(test_data)

intermediateImage = np.reshape(list(lTemp1)[0][0],(7,7))
plt.imsave('inter.png',intermediateImage)

net.SGD(training_data, 30, 50, 2, test_data=test_data)

beforeImage = np.reshape(list(lTemp)[0][1],(28,28))
plt.imsave('before.png',beforeImage)


after = net.feedforward(list(lTemp2)[0][0])

afterImage = np.reshape(after,(28,28))
plt.imsave('after.png',afterImage)
