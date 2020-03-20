import numpy as np

N = 1000
in_shape = 64
hidden_shape = 100
out_shape = 10
lr = 1e-6
epochs = 300


x = np.random.randn(N, in_shape).astype(np.int32)
y = np.random.randn(N, out_shape).astype(np.int32)

w1 = np.random.randn(in_shape, hidden_shape)
b1 = np.random.randn(hidden_shape)
w2 = np.random.randn(hidden_shape, out_shape)
b2 = np.random.randn(out_shape)


for epoch in range(epochs):
  # Forward Pass
  h = x @ w1 + b1
  h_relu = h * (h > 0)
  h2 = h_relu @ w2 + b2
  y_pred = h2 * (h2 > 0)
  
  e = y_pred - y
  loss = np.sum(e ** 2)
  print(f'Epoch {epoch+1} / {epochs}\tLoss {loss}')

  # Gradient w.r.t loss
  grad_y_pred = 2. * e

  # Gradient w.r.t ReLU
  grad_h2 = grad_y_pred * (grad_y_pred > 0)

  # Parameter gradients
  grad_w2 = h_relu.T @ grad_h2
  grad_b2 = np.sum(grad_h2, axis=0)

  # Gradient w.r.t second layer
  grad_h1_relu = grad_h2 @ w2.T

  # Gradient w.r.t ReLU
  grad_h1 = grad_h1_relu * (grad_h1_relu > 0)

  # Parameter gradients
  grad_w1 = x.T @ grad_h1
  grad_b1 = np.sum(grad_h1, axis=0)

  # Update parameters
  w1 -= lr * grad_w1
  b1 -= lr * grad_b1
  w2 -= lr * grad_w2
  b2 -= lr * grad_b2
  
