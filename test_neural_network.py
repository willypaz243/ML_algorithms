import unittest
import numpy as np
from neural_network import Dense, Layer, Model


def testing_model_outputs(model: Model, condition):
    examples: int = np.random.randint(512) + 1  # random number of examples between 1-512
    inputs: int = np.random.randint(256) + 1  # random number of inputs between 1-256
    X = np.random.rand(examples, inputs)
    Y = model(X)
    assertion = condition(X, Y)
    return assertion


def add_many_layers_to_model(model: Model, layer:Layer):
    layers = np.random.randint(10) + 1  # random number of layers between 1-10
    for _ in range(layers):
        neurons = np.random.randint(128) + 1
        model.add(layer(neurons))


class TestNeuralNetwork(unittest.TestCase):
    def test_model_without_layers(self):
        """
        The model has no intermediate layers, the input and output must be the same 
        because the data was processed
        """
        assert testing_model_outputs(
            Model([]),
            # input and output are the same, so their subtraction is 0
            condition=lambda X, Y: np.mean(Y - X) == 0
        )

    def test_model_number_of_examples_constant(self):
        """
        The input and the output are different but must have the same number of examples.
        If the input is from (m, i) the output must be from (m, n)
        """
        model = Model([])
        add_many_layers_to_model(model, Layer) # let's pass the Layer class as an argument
        assert testing_model_outputs(
            model,
            # X.shape[0] is the number of examples
            condition=lambda X, Y: Y.shape[0] == (X.shape[0])
        )

    def test_model_output_shape(self):
        """
        if you have an input with the form (m, i) and the last layer has "n" neurons, 
        the output must have the form (m, n)
        """
        model = Model([])
        add_many_layers_to_model(model, Layer) # let's pass the Layer class as an argument
        last_layer_neurons = 3
        model.add(Layer(last_layer_neurons))
        assert testing_model_outputs(
            model,
            # X.shape[0] is the number of examples of the input
            condition=lambda X, Y: Y.shape == (X.shape[0], last_layer_neurons)
        )


if __name__ == "__main__":
    unittest.main()
