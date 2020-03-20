import mikenet
import numpy as np

in_shape = 2
hidden_shape = 5
out_shape = 2

# Training set
x = np.array([
  [0, 0],
  [0, 1], 
  [1, 0],
  [1, 1]
])

y = np.array([
  [1, 0],
  [0, 1],
  [0, 1],
  [1, 0]
])

model = mikenet.Sequential(
  mikenet.Linear(in_shape, hidden_shape),
  mikenet.Tanh(),
  mikenet.Linear(hidden_shape, out_shape),
)

mikenet.train(x, y, model, epochs=5000)

# Evaluate
print()

print(f'x = {x}')
print(f'y = {y}')
print(f'preds = {model(x)}')
