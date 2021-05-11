import time
import numpy as np

from neural_network import Model


def mse(Y, target):
    return np.mean((target - Y) ** 2, axis=1)


def d_mse(Y, target):
    return (2 * (Y - target)) / Y.shape[1]


def train_step(model: Model, X: np.ndarray, target: np.ndarray, lr: float):
    """
    Stochastic gradient descent

    Parameters
    ----------
    model : Model
        This is the model for train

    X : np.ndarray
        This is the input data

    target : np.ndarray
        This is the target output

    fault : np.ndarray
        this is a part of the error caused by each neuron

    Returns
    -------
    np.ndarray
        The model error in this step.
    np.ndarray
        The accuracy
    """
    Y = model(X, training=True)  # model prediction
    acc = np.int64(target.argmax(axis=1) ==
                   Y.argmax(axis=1)).mean()  # accuracy
    loss: np.ndarray = mse(Y, target)  # prediction loss
    gradients = []  # gradients
    for layer in reversed(model.layers):  # Back-propagation
        derivative = layer.activation['derivative']  # activation derivative
        if layer == model.layers[-1]:
            fault: np.ndarray = d_mse(Y, target)
        # calculating the fault of the neuron
        fault = fault * derivative(layer.Y)
        # Gradient calculation
        gradient: np.ndarray = np.matmul(fault.T, layer.X).T
        b_gradient: np.ndarray = np.matmul(
            fault.T, np.ones((X.shape[0], 1))).T
        gradients.append((gradient, b_gradient))  # save to gradients
        # update fault for back layer
        fault: np.ndarray = np.matmul(fault, layer.W.T)
    update_parameters(model, gradients, lr)

    return loss.mean(), acc


def update_parameters(model: Model, gradients: list, lr: float):
    for layer, gradient in zip(reversed(model.layers), gradients):
        gradient, b_gradient = gradient
        layer.W = layer.W - lr * gradient
        layer.B = layer.B - lr * b_gradient


def fit(model: Model, X: np.ndarray, targets: np.ndarray,
        batch_size: int, epochs: int, learning_rate: float = 0.0001):
    """
    ## Parameters 

    `X`:`np.ndarray`
        - Is a matrix with the shape (m,i) contains the examples for the training of the model.

    `targets`:`np.ndarray`
        - are the expected outputs with the shape (m, n), `n` is the number of the outputs.

    `batch_size`:`int`
        - (Positive number) Is the size of the batch in which the examples are grouped, the data can be grouped in small batches 
        to reduce the computational cost of training and help the model to generalize the information, 
        if its value is `1` it will be trained the model sequentially.
    
    `epochs`:`int`
        - (Positive number) Is the number of times the model will cycle through all the examples

    `learning_rate`:`float`
        - (Positive number) It is the size of the step of modification of the weights towards the point of 
        least loss when the model is training.
    """
    # X and targets must have the same number of examples
    assert X.shape[0] == targets.shape[0]
    for epoch in range(epochs):

        last_examples = len(X) % batch_size
        # batching the inputs
        x = list(X[last_examples:].reshape(
            len(X) // batch_size, -1, X.shape[-1]))
        if X[:last_examples].size > 0:
            x.append(X[:last_examples])
        # batching the targets
        target = list(targets[last_examples:].reshape(
            len(X) // batch_size, -1, targets.shape[-1]))
        target.append(targets[:last_examples])
        # metrics variables
        error = None
        accuracy = None
        start = time.time()
        # epoch iteration
        print(f'Epoch: {epoch+1}:')
        for i, inp, tar in zip(range(len(x)), x, target):
            loss, acc = train_step(model, inp, tar, learning_rate)
            # metrics calculation
            if error is None and accuracy is None:
                error = loss
                accuracy = acc
            else:
                error = (error + loss) / 2
                accuracy = (accuracy + acc) / 2
            # printing metrics
            msg = f'batchs: {i+1}/{len(x)}, Error: {error} Acc: {acc}'
            print("", end='\r')
            print(msg, end='')
        print("", end='\r')
        print(msg + f' Time: {time.time() - start}')
