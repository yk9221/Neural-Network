# Neural Network from Scratch
This project implements a neural network with custom layers and optimizers from scratch in Python using NumPy.

## Installation
To get started, install the necessary dependencies:  
```bash
pip install numpy
pip install matplotlib
```

## Forward Propagation
In forward propagation, the input data passes through each layer, applying linear transformations, activations, and loss computations.

### Linear Layer:
The linear layer performs the following transformation:  
$$s = W^{T}x + b$$  
Where:
- $$W$$ is the weight matrix.
- $$x$$ is the input vector.
- $$b$$ is the bias vector.

```python
class Linear():
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size) * np.sqrt(2 / (in_size + 1))
        self.b = np.zeros((out_size, 1))
    
    def forward(self, x):
        self.x = x
        self.s = np.dot(np.transpose(self.W), self.x) + self.b
        return self.s
```
The `Linear` class represents a fully connected layer that performs a linear transformation. Weight `W` is initialized using Xavier initialization.

### ReLU Activation:
`ReLU` (Rectified Linear Unit) applies the following operation element-wise:  
$$f(s) = max(0, s)$$

```python
class ReLU():
    def forward(self, s):
        self.s = s
        return np.maximum(0, s)
```
The `ReLU` activation function applies element-wise non-linearity, setting negative values to zero.

### Tanh Activation:
The `Tanh` activation applies the hyperbolic tangent function element-wise:  
$$f(s) = tanh(s)$$  

```python
class Tanh():
    def forward(self, s):
        self.s = s
        self.tanh = np.tanh(s)
        return self.tanh
```
The `Tanh` activation function is useful in cases where the model needs to output both negative and positive values.

### Sigmoid Activation:
The `Sigmoid` activation function applies the following operation element-wise:  
$$f(s) = \frac{1}{1 + e^{-s}}$$  
```python
class Sigmoid():
    def forward(self, s):
        self.s = s
        self.sigmoid = 1 / (1 + np.exp(-s))
        return self.sigmoid
```
The `Sigmoid` activation function is often used in binary classification problems, as it outputs probabilities.  

### Softmax Activation:
The softmax function is applied to the output layer to normalize the logits into probabilities. The softmax  function is defined as:  
$$\sigma(s)_{i} = \frac{e^{s_i}}{\sum e^{s_j}}$$  
This ensures that the output values sum to 1, representing a probability distribution.

```python
class Softmax():    
    def forward(self, s):
        e_s = np.exp(s - np.max(s, axis=0, keepdims=True))
        self.output = e_s / e_s.sum(axis=0, keepdims=True)
        return self.output
```
The `Softmax` class normalizes the output scores to represent probabilities.

### Cross Entropy Loss:
The cross-entropy loss for multi-class classification is defined as:  
$$L = \sum_{i = 1}^{N}\sum_{j = 1}^{C}y_{ij}\log(\hat{y}_{ij})$$  
Where:
- $$y$$ is the true label (one-hot encoded).
- $$\hat{y}$$ is the predicted probability from softmax.
- $$N$$ is the number of samples.
- $$C$$ is the number of classes.

```python
class CrossEntropyLoss():
    def forward(self, x_L, y):
        epsilon = 1e-12
        x_L = np.clip(x_L, epsilon, 1 - epsilon)
        self.x_L = x_L
        self.y = y
        self.N = x_L.shape[1]
        return -np.sum(y * np.log(x_L)) / self.N
```
The `CrossEntropyLoss` function calculates the loss between predicted probabilities and true labels, helping measure the error for classification tasks.

### Binary Cross Entropy Loss:
For binary classification, the loss is:  
$$L = -\frac{1}{N}\sum_{i = 1}^{N}\big[ y_{i}\log(\hat{y_{i}}) + (1 - y_{i})\log(1 - \hat{y_{i}}) \big]$$  
Where:
- $$y$$ is the true label (one-hot encoded).
- $$\hat{y}$$ is the predicted probability from softmax.
- $$N$$ is the number of samples.

```python
class BinaryCrossEntropyLoss():
    def forward(self, x_L, y):
        epsilon = 1e-12
        x_L = np.clip(x_L, epsilon, 1 - epsilon)
        self.x_L = x_L
        self.y = y
        self.N = x_L.shape[1]
        return -np.sum(y * np.log(x_L) + (1 - y) * np.log(1 - x_L)) / self.N
```
The `BinaryCrossEntropyLoss` function loss is suitable for problems with two classes.

### Mean Squared Error:
`MSELoss` is commonly used in regression tasks and is defined as:  
$$L = \frac{1}{N}\sum_{i = 1}^{N}(x_{L} - y)^{2}$$  
Where:
- $$x_{L}$$ is the predicted output.
- $$y$$ is the true label.

```python
class MSELoss():
    def forward(self, x_L, y):
        self.x_L = x_L
        self.y = y
        self.N = x_L.shape[1]
        return np.sum((x_L - y) ** 2) / self.N
```

## Back Propagation
Backward propagation involves calculating the gradients of the loss with respect to each parameter using the chain rule.

### Linear Layer Backward:
The gradients for the linear layer are computed as:  
$$\frac{\partial L}{\partial W} = x \cdot \delta^{T}$$  
$$\frac{\partial L}{\partial b} = \sum \delta$$  
Where:
- $$\delta$$ is the error propagated backward from the next layer.

```python
class Linear():
    def backward(self, gradient):
        self.dW = np.dot(self.x, np.transpose(gradient))
        self.db = np.sum(gradient, axis=1, keepdims=True)
        return np.dot(self.W, gradient)
```
The `Linear` class computes gradients for the weights and biases, passing the error backward through the network.

### ReLU Backward:
The gradient for `ReLU` is:  
$$\frac{\partial L}{\partial s} = \delta \cdot I(s > 0)$$  
Where:
- $$I(s > 0)$$ is an indicator function that is $$1$$ if $$s > 0$$, and $$0$$ otherwise.

```python
class ReLU():
    def backward(self, gradient):
        return gradient * (self.s > 0).astype(float)
```

### Tanh Backward:
For tanh, the gradient is:  
$$\frac{\partial L}{\partial s} = \delta \cdot (1 - tanh(s)^{2})$$  
Where:
- $$tanh(s)$$ is the hyperbolic tangent function.

```python
class Tanh():
    def backward(self, gradient):
        return gradient * (1 - self.tanh ** 2)
```

### Sigmoid Backward:
For `sigmoid`, the gradient is:  
$$\frac{\partial L}{\partial s} = \delta \cdot \sigma(s) \cdot (1 - \sigma(s))$$  
Where:
- $$\sigma(s)$$ is the sigmoid function.

```python
class Sigmoid():
    def backward(self, gradient):
        return gradient * (self.sigmoid * (1 - self.sigmoid))
```

### Softmax Backward:
```python
class Softmax():
    def backward(self, gradient):
        return gradient
```
The `Softmax` backward pass is straightforward since the gradient is propagated directly.

### Cross Entropy Loss Backward:
For `CrossEntropyLoss`, the gradient with respect to the input $$s$$ is:  
$$\frac{\partial L}{\partial s} = \hat{y} - y$$

```python
class CrossEntropyLoss():
    def backward(self):
        return (self.x_L - self.y) / self.N
```

### Binary Cross Entropy Loss Backward:
For `BinaryCrossEntropyLoss`, the gradient with respect to the input $$s$$ is:  
$$\frac{\partial L}{\partial s} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

```python
class BinaryCrossEntropyLoss():
    def backward(self):
        return (self.x_L - self.y) / (self.x_L * (1 - self.x_L) * self.N)
```

### Mean Squared Error Backward:
For `MSELoss`, the gradient with respect to the input $$s$$ is:  
$$\frac{\partial L}{\partial s} = 2(x_{L} - y)$$

```python
class MSELoss():
    def backward(self):
        return 2 * (self.x_L - self.y) / self.N
```

## Optimizers
Optimizers update the weights of the model during training, based on the gradients computed during backpropagation.

### Basic Optimizer:
The basic optimizer implements standard stochastic gradient descent (SGD):  
$$W = W - \eta\nabla L(W)$$  
Where:
- $$\nabla L(W)$$ is the gradient of the loss function with respect to the weights.
- $$\eta$$ is the learning rate.
```python
class BasicOptimizer(Optimizer):
    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, dW, db, layer):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db
```

### Momentum Optimizer
The `momentum optimizer` improves gradient descent by accumulating an exponentially decaying moving average of past gradients:  
$$v_{t} = \beta v_{t - 1} + (1 - \beta) \nabla L(W)$$  
$$W = W - \eta v_{t}$$  
Where:  
- $$\beta$$ is the momentum coefficient.
- $$\nabla L(W)$$ is the gradient of the loss function with respect to the weights.
- $$v$$ is the velocity (or accumulated gradient).
- $$\eta$$ is the learning rate.

### Adam Optimizer
`Adam` combines the advantages of momentum and adaptive learning rates using two moving averages: one for the gradients and another for the squared gradients. The update rule is given by:  
$$m_{t} = \beta_{1}m_{t - 1} + (1 - \beta{1})\nabla L(W)$$  
$$v_{t} = \beta_{2}v_{t - 1} + (1 - \beta{2})(\nabla L(W))^{2}$$  
$$\hat{m_{t}} = \frac{m_{t}}{1 - \beta_{1}^{t}}$$  
$$\hat{v_{t}} = \frac{v_{t}}{1 - \beta_{2}^{t}}$$  
$$W = W - \eta\frac{\hat{m_{t}}}{\sqrt{\hat{v_{t}}} + \epsilon}$$  
Where:
- $$m_{t}$$ and $$v_t$$ are moving averages of the gradient and squared gradient.
- $$\beta_{1}$$ and $$\beta_{2}$$ are decay rates.
- $$\epsilon$$ is a small constant for numerical stability.

## Neural Network Architecture
The `NN` class defines the architecture of the neural network, managing layers and handling the forward and backward propagation processes. Key functionalities include:
- **Adding Layers:** Layers can be added to the network using a sequential method, allowing for flexible architecture definition.
- **Forward Propagation:** The `forward` method processes input through each layer in sequence, producing an output.
- **Backward Propagation:** The `backward` method computes gradients starting from the output layer and propagates them back through the network.
- **Weight Updates:** After calculating gradients, the `update_weights` method utilizes the chosen optimizer to adjust the weights.

## Dataset
The dataset used for training and evaluation is the MNIST dataset, consisting of handwritten digits. It comprises:
- **60,000 training samples:** Used to train the model.
- **10,000 test samples:** Used to evaluate the model's performance and generalization capability.

The MNIST dataset is a widely recognized benchmark in the field of machine learning and is commonly used for training image processing systems.

## Training and Defining the Architecture
The training process involves several steps:
- **Shuffling Data:** The training data is shuffled at the beginning of each epoch to ensure that the model does not learn from patterns based on the order of data.
- **Mini-Batching:** The data is divided into mini-batches to allow for more efficient training and to stabilize the gradient updates.
- **Loss Computation and Weight Updates:** For each batch, the loss is computed, backpropagation is performed, and weights are updated accordingly.

The architecture of the model is defined using the `NN` class and can consist of various layers such as flattening, linear layers, activation functions, and softmax.

## Results
After training the model, it was evaluated on the test dataset. The performance metrics were as follows:
- **Test Accuracy:** The final test accuracy achieved was **98.42%**, demonstrating the model's high capability to generalize well to unseen data.
- **Final Loss:** The loss on the test set was measured at **0.0021**, reflecting a low error rate and confirming that the model has learned to classify the data accurately.

These results highlight the effectiveness of the implemented neural network architecture and the training process.