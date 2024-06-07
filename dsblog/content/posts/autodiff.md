+++
title = 'Reverse-mode autodiff from scratch'
date = 2024-06-07
math = true
+++

We implement a simple automatic differentiation tool in Python which can
compute the gradient of any (simple) multivariable function efficiently.

## Use case
Understanding how autodiff works is crucial for understanding backpropagation
and how optimisation works in a deep learning setting: In general, we want an
easy way to compute gradients of a loss function wrt to its weights and bias
parameters so that we can apply algorithms such as gradient descent.

In a typical ML optimisation setting, we have some loss function $L$, parameters $W$ and learning rate $\eta$:  

\begin{equation}
W_{k+1} = W_k - \eta \frac{dL}{dW}
\end{equation}

In practice, we rely on automatic differentiation libraries such as
[JAX](https://jax.readthedocs.io/en/latest/quickstart.html) to handle this, but
it is useful to understand the underlying logic behind this.

## Backpropagation Calculus
The fundamental idea behind autodiff is to represent a function's expression in
a directed acyclic graph, where nodes represent variables and edges represent
partial derivatives from mathematical operations like addition, multiplication, exp, log, etc.

Consider a function:

\begin{equation}
L(x,y) = x \times y = (a-b) \times (b+1)
\end{equation}

We can represent this as a computational graph as follows:
![Computational Graph](/img/Computational_Graph.png#center)

Using the chain rule, we can multiply partial derivatives to obtain the
derivative of $L$ wrt to any variable: This is the foundation of
backpropagation.

$$
\frac{dL}{da} = \frac{dL}{dx} \frac{dx}{da} = y \cdot 1
$$

## Implementation
With this understanding, it is simple to implement this idea in code. We first
define a `Variable` class which stores its own value, as well as
pointers to its child nodes and its respective local derivatives. 

```python
class Variable:
    def __init__(self, val, children=(), local_gradients=()):
        self.val = val
        self.children = children
        self.local_gradients = local_gradients
        self.name = ""

    def set_name(self, name) -> None:
        self.name = name
```

We also need to polymorph the object's basic mathematical operations like
addition to return another `Variable` object, while storing local derivatives.
For instance, we override the addition behaviour as such:

```python
class Variable:
    ...    

    def __add__(self, other: Variable) -> Variable:
        out = Variable(
                val = self.val + other.val,
                children = (self, other),
                local_gradients = (
                    (self, 1),
                    (other, 1),
                )
              ).set_name(f"{self.name} + {other.name}")
        return out
```

## Training a neural network
Let's test our autodiff implementation by training a neural network from scratch. We simulate 100 samples of input and output data from
a `Unif(0,1)` distribution, and use MSE as our loss function:
$$
L(\hat{y}, y) = \frac{1}{n} \sum^n_{i=1} (\hat{y} - y)^2
$$

$$
X \in \mathbb{R}^{100,50}, y \in \mathbb{R}^{100}
$$

For simplicity, we use a single hidden layer, and our network is defined as:
$$
W \in \mathbb{R}^{50,1}, b \in \mathbb{R}^{100}
$$
$$
\hat{y} = XW + b
$$

We also use standard gradient descent, and we have 51 parameters to optimise:
$$
W_{ij}^{(k+1)} = W_{ij}^{(k)} - \eta \frac{dL}{dW_{ij}}
$$
$$
b_{i}^{(k+1)} = b_{i}^{(k)} - \eta \frac{dL}{db_{i}}
$$

The below figure shows the loss decreasing over epochs, indicating that our
gradient computations are indeed correct.

![Train Loss](/img/Train_Loss.png#center)

## Conclusions
We have successfully implemented simple automatic differentiation from scratch
by representing variables in an expression tree. While the concept seems
trivial, it is pretty key to understanding how optimisation in machine learning
works.