import numpy as np

class Loss:

  def __call__(self, predicted, actual):
    raise NotImplementedError

  def grad(self, predicted, actual):
    raise NotImplementedError

class MSE(Loss):
  
  def __call__(self, predicted, actual):
    return np.sum((predicted - actual) ** 2)

  def grad(self, predicted, actual):
    return 2 * (predicted - actual)

