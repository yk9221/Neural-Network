# Neural Network from Scratch
## Basic Notation
Input layer: $$l = 0$$  
Hidden layers: $$1 \leq l \leq L - 1$$  
Output layer: $$l = L$$  
$$d^{(l)}$$: Number of nodes in layer $$l$$, $$0 \leq l \leq L - 1$$  (excluding bias)  
$$\underline{x}^{(l)} = [1 \quad x_{1}^{(l)} \quad ... \quad x_{d^{(l)}}^{(l)}]^T$$: The value of each neuron at layer $$l$$  
$$w_{i, j}^{(l)}$$: Weight connecting node $$i$$ in layer $$l - 1$$ to node $$j$$ in layer $$l$$  
$$\theta(s)$$: Activation function  
$$\Omega=\big[W^{(1)} \quad W^{(2)} \quad ... \quad W^{(L)}\big]$$: Weights from layer $$1$$ to layer $$L$$  
$$e(\Omega)=g(x^{(L)}, y)$$: Error between the predicted value and the real value  
$$\delta^{(l)}$$: Sensitivity of the error, $$e(\Omega)$$, with respect to the sum $$\underline{s}^{(l)}$$  

## Front Propagation
The input is given in the form: $$\underline{x}^{(0)} = [1 \quad x_1 \quad x_2 \quad ... \quad x_d]^T$$  
Then for each layer, $$l$$, the sum and the value of the next neuron is calculated with the following equations.  
$$\underline{s}^{(l)} = \big(W^{(l)}\big)^T \underline{x}^{(l - 1)}$$  
$$\underline{x}^{(l)} = \big[1 \quad \theta(\underline{s}^{(l)})\big]^T$$  
After looping through all the layers the front propagation function returns the following matrices.  
$$X = \big[\underline{x}^{(0)} \quad \underline{x}^{(1)} \quad ... \quad \underline{x}^{(L)}\big]$$  
$$S = \big[\underline{s}^{(1)} \quad \underline{s}^{(2)} \quad ... \quad \underline{s}^{(L)}\big]$$  

## Back Propagation
The sensitivity of the last layer $$l = L$$ can be calculated as follows.  
$$\underline{\delta}^{(L)} = \frac{\partial e(\Omega)}{\partial \underline{s}^{(L)}} = \frac{\partial g(x^{(L)}, y)}{\partial \underline{x}^{(L)}} \cdot \theta'(\underline{s}^{(L)})$$  
For layers $$0 < l < L$$, the following equation can be used to calculate the sensitivity.  
$$\underline{\delta}^{(l)} = [\hat{W}^{(l + 1)} \underline{\delta}^{(l + 1)}] \otimes \theta'(\underline{s}^{(L)})$$  
Then the gradient of the error with respect to the weight can be calculated as follows.  
$$\frac{\partial e(\Omega)}{\partial w^{(l)}}=\underline{x}^{(l - 1)}(\underline{\delta}^{(l)})^T$$  
The back propagation function will return the following matrix.  
$$g = \big[ \frac{\partial e(\Omega)}{\partial w^{(1)}} \quad \frac{\partial e(\Omega)}{\partial w^{(2)}} \quad ... \quad \frac{\partial e(\Omega)}{\partial w^{(L)}} \big]$$  

## Updating the Weights
After calculating the gradients for each layer, the new weights can be calculated.  
$$W^{(l)}(t + 1) = W^{(l)}(t) - \eta \frac{\partial e(\Omega)}{\partial W^{(l)}}$$  

## Activation Function
The ReLU function was used for the activation function which is defined as  
$$\theta(s) = max(0, s)$$

## Output Function
The sigmoid function was used for the output function which is defined as  
$$\sigma(s) = \frac{1}{1+e^{-s}}$$

## Error Function
The log loss function was used for the error function which is defined as  
$$g(x^{(L)}, y)=-(y \log(x^{(L)}) + (1 - y) \log(1 - x^{(L)}))$$