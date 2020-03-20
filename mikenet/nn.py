class NeuralNet:
  '''
  Generic neural network to handle forward
    and backward propogation for all layers in network.
  '''
  
  def __call__(self, inputs):
    raise NotImplementedError

  def backward(self, grads):
    raise NotImplementedError


class Sequential(NeuralNet):

  def __init__(self, *layers):
    self.layers = layers

  def __call__(self, inputs):
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs

  def backward(self, grads):
    for layer in reversed(self.layers):
      grads = layer.backward(grads)
    return grads

  def params_and_grads(self):
    for layer in self.layers:
      for name, weights in layer.params.items():
        grads = layer.grads[name]
        yield weights, grads

  def __len__(self):
    return len(self.layers)
