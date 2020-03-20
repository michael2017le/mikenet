import numpy as np


class Layer:
  '''
  Generic layer class that handles
    forward and backward propogation
    for calculating gradients
  '''
  
  def __init__(self):
    self.params = {}
    self.grads = {}

  def __call__(self, inputs):
    '''
    param inputs: matrix inputs from previous layer
    '''
    raise NotImplementedError

  def backward(self, grads):
    '''
    param grads: matrix gradients calculated for backpropogation
    '''
    raise NotImplementedError

class Linear(Layer):
  '''
  Constructs linear, fully-connected layer
    with weights and bias parameters.
  '''

  def __init__(self, in_shape, out_shape, initialize=None):
    super().__init__()
    self.params['w'] = np.random.randn(in_shape, out_shape)
    self.params['b'] = np.random.randn(out_shape)

  def __call__(self, inputs):
    ''' 
    Simple matrix math with weights and bias
      and store inputs for backpropagation

    output = input @ weight + bias
    '''
    self.inputs = inputs
    return inputs @ self.params['w'] + self.params['b']

  def backward(self, grads):
    '''
    Gradients for weights and bias parameters
      are calculated with respect to the gradients passed
      down from next layer in network.

    i is input from previous layer
    x is this layer's output
    f is activation function applied to x
    w and b are weights and bias respectively
      for this layer's parameters

    y = f(x)
    x = i @ w + b
    dy/di = f'(x) @ w.T
    dy/dw = i.T @ f'(x)
    dy/db = f'(x)
    '''

    self.grads['w'] = self.inputs.T @ grads
    self.grads['b'] = np.sum(grads, axis=0)
    return grads @ self.params['w'].T

class Activation(Layer):
  '''
  Generic activation layer that simply applies
  a function to all values in the input.
  '''

  def __init__(self, f, f_prime):
    super().__init__()
    self.f = f
    self.f_prime = f_prime

  def __call__(self, inputs):
    self.inputs = inputs
    return self.f(inputs)

  def backward(self, grads):
    '''
    Uses chain rule to calculate gradients
    to the previous layer.

    x is inputs from previous layer f is activation function for this layer
    
    y = f(x) and x = g(z)
    dy/dz = f'(x) * g'(z)
    '''

    return self.f_prime(self.inputs) * grads


def _tanh(x):
  return np.tanh(x)

def _tanh_prime(x):
  y = np.tanh(x)
  return 1 - y ** 2

class Tanh(Activation):
  def __init__(self):
    super().__init__(_tanh, _tanh_prime) 


def _sigmoid(x):
  return 1. / (1 + np.exp(-x))

def _sigmoid_prime(x):
  return x - x ** 2

class Sigmoid(Activation):
  def __init__(self):
    super().__init__(_sigmoid, _sigmoid_prime)


def _leaky_relu(alpha):
  def f(x):
    pos_x = np.maximum(x, 0)
    neg_x = x * alpha * (x <= 0)
    return pos_x + neg_x
  return f

def _leaky_relu_prime(alpha):
  def f(x):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
  return f

class LeakyReLU(Activation):
  def __init__(self, alpha=0.1):
    super().__init__(_leaky_relu(alpha), _leaky_relu_prime(alpha))


class ReLU(LeakyReLU):
  '''
  Special case of LeakyReLU where alpha = 0
  '''
  def __init__(self):
    super().__init__(alpha=0.)


