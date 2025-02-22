import numpy as np
import matplotlib.pyplot as plt

# we assume it is for linear regression we will use MSE loss function

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def ReLU_deriv(Z):
    return (Z > 0).astype(float)

def linear(x):
    return x

def MSE(X, Y):
    return np.mean((X - Y) ** 2)

def MSE_deriv(X, Y):
    return (X - Y)

ACTIVATIONS = {
    "relu": relu,
    "linear": linear
}

class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = layers

    def forward_prop(self, X):
        results = {}
        for count, layer in enumerate(self.layers, start=0):
            Z, A = layer.forward(X)
            results[count] = {'Z': Z, 'A': A}
            X = A  # Next input will be the activation of this layer
        return A, results

    def fit(self, X, Y, alpha, iterations):
        Y = np.array(Y, dtype=np.float32)
        X = np.array(X, dtype=np.float32).T  # Ensure X is a numpy array (i x 1)
        for _ in range(iterations):
            A, results = self.forward_prop(X)
            gradient = self.backward_prop(X,Y, results)
            self.update_weights(gradient, alpha)
            print(f"Iteration {_+1}, Loss: {MSE(A, Y)}")
        return MSE(self.forward_prop(X)[0], Y)

    def backward_prop(self, X, Y, results):
        gradient = {}  # Dictionary to store gradients
        start = len(self.layers) - 1
        m = Y.shape[1]  # Number of examples
        # Compute the output layer gradients
        dA = (2 / m) * (results[start]['A'] - Y)  # MSE derivative
        dZ = dA  # Assuming linear activation at output layer
        dW = np.dot(dZ, results[start-1]['A'].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m  # Average over batch
        gradient[start] = {'dW': dW, 'db': db}

        dZ_prv = dZ  # Store for backpropagation

        # Iterate over hidden layers (from last hidden to first hidden)
        for count in range(start-1, -1, -1):  # Fixed range to include 0
            dA = np.dot(self.layers[count + 1].weights.T, dZ_prv)  # Propagate error backward
            dZ = dA * ReLU_deriv(results[count]['Z'])  # Apply ReLU derivative (should be on Z, not A)
            dW = np.dot(dZ, (results[count-1]['A'].T if count > 0 else X.T)) / m  # Use X for first layer
            db = np.sum(dZ, axis=1, keepdims=True) / m  # Sum across batch

            gradient[count] = {'dW': dW, 'db': db}
            dZ_prv = dZ  # Store for next layer

        return gradient

    def update_weights(self, gradient, alpha):
        for count, layer in enumerate(self.layers, start=0):
            layer.weights -= alpha * gradient[count]['dW']
            layer.bias -= alpha * gradient[count]['db']

class Layer:
    def __init__(self, neurons_num, input_size, activation="relu"):
        self.neurons_num = neurons_num
        self.activation_func = ACTIVATIONS.get(activation, relu)
        self.weights = np.random.randn(neurons_num, input_size)
        self.bias = np.zeros((neurons_num, 1))

    def forward(self, X):
        Z = np.dot(self.weights, X) + self.bias
        A = self.activation_func(Z)
        return Z, A



model = NeuralNetwork([
    Layer(10, input_size=1, activation="relu"),
    Layer(64, 10, activation="relu"),
    Layer(6, input_size=64, activation="relu"),
    Layer(1, input_size=6, activation="linear")
])

# Input should be an array
x_train = np.linspace(-10, 10, 10000).reshape(-1, 1)  # 100 points between -2 and 2
y_train = np.exp(x_train).T
# print(y_train.shape)
print("results", model.fit(x_train, y_train, 0.8, 10000))

y_pred = model.forward_prop(x_train.T)[0]

plt.scatter(x_train, y_train.T, label='True Function', color='blue')
plt.plot(x_train, y_pred.T, label='NN Approximation', color='red')
plt.legend()
plt.savefig("nn_x7_plot.png")
