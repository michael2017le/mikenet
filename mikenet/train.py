import numpy as np
from mikenet.loss import MSE
from mikenet.optim import SGD

class BatchIterator:

  def __init__(self, batch_size=32, shuffle=True):
    self.batch_size = batch_size
    self.shuffle = shuffle

  def __call__(self, x, y):
    idx = np.arange(0, len(x), self.batch_size)

    if self.shuffle:
      np.random.shuffle(idx)

    for i in idx:
      end_i = i + self.batch_size
      yield x[i : end_i], y[i : end_i]

def train(x, y, nn, loss=MSE(), optim=SGD(), batch_iterator=BatchIterator(), epochs=100):

  for epoch in range(epochs):
    total_loss = 0.
    for inputs, actual in batch_iterator(x, y):
      predicted = nn(inputs)
      total_loss += loss(predicted, actual)
      grads = loss.grad(predicted, actual)
      nn.backward(grads)
      optim.step(nn)

    print(f'Epoch: {epoch+1} / {epochs}\tLoss: {total_loss}')
