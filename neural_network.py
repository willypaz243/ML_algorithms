import numpy as np

# a dict for select a activation function
activations = {
    'sigmoid': {
        'function': lambda X: 1 / (1 + np.e**(-X)),
        'derivative': lambda X: np.e**(-X) / ((1 + np.e**(-X))**2)
    },
    'tanh': {
        'function': lambda X: np.tanh(X),
        'derivative': lambda X: 1 - np.tanh(X) ** 2
    },
    'relu': {
        'function': lambda X: ((X / (X.max() + 1)) * (X > 0)),
        'derivative': lambda X: 1 * (X >= 0)
    }
}


class Layer:
    def __init__(self, neurons):
        self.neurons:int = neurons # the layer neurons
        self.shape:tuple = None # input shape
        self.W = None # the wight matrix, shape (inputs, neurons)
        self.B = None # the bias matrix,  shape (1, neurons)

    def forward(self, X): # output shape (input_examples, neurons)
        if self.shape is None:
            self.init_layer(X)
        return np.matmul(X, self.W) + self.B # matrix multiplication

    def init_layer(self, X):
        self.shape = (X.shape[-1], self.neurons) # X.shape (input_examples, input_data_size)
        # create initial parameters
        self.W = np.random.rand(*self.shape) * 2 - 1
        self.B = np.random.rand(1, self.neurons) * 2 - 1

    def __call__(self, X, training=False):
        Y = self.forward(X)
        if training: # if the model is training
            self.X = X
            self.Y = Y
        return Y


class Dense(Layer): # this inherits from the Layer class
    def __init__(self, neurons, activation='sigmoid'): # sigmoid activations for default
        super().__init__(neurons) # init the layer 
        self.activation:dict = activations[activation] # get specific activations functions

    def forward(self, X):
        Y = super().forward(X)
        return self.activation['function'](Y) # calculate with activations function


class Model:
    def __init__(self, layers:list=[]):
        self.layers:list = layers

    def add(self, layer: Layer):# add a Layer
        self.layers.append(layer)

    def forward(self, X, training=False): # forward in all layers
        Y = X
        for layer in self.layers:
            Y = layer(Y, training)
        return Y

    def __call__(self, X, training=False):
        return self.forward(X, training)
