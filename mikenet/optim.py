class Optimizer:

  def step(self, nn):
    raise NotImplementedError

class SGD(Optimizer):

  def __init__(self, lr=1e-3):
    self.lr = lr

  def step(self, nn):
    for param, grad in nn.params_and_grads():
      param -= self.lr * grad
    
