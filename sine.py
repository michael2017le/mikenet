import mikenet
import numpy as np

N = 1000
hidden_dim = 100
input_dim = 1
output_dim = 1

# Training data
x = np.random.randn(N, 1).astype(np.int32)
y = np.sin(x)

model = mikenet.Sequential(
  mikenet.Linear(input_dim, hidden_dim),
  mikenet.ReLU(),
  mikenet.Linear(hidden_dim, output_dim)
)

# This by default uses stochastic gradient descent
#  using the mean squared error loss function
mikenet.train(x, y, model)

print()

# Testing set
loss = mikenet.MSE()
test_size = int(.3 * N)

test_x = x[-test_size:]
test_y = y[-test_size:]

preds = model(test_x)

test_loss = loss(preds, test_y)
print(f'Test loss: {test_loss}')

