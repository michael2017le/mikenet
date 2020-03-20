import mikenet
import numpy as np

N = 1000
in_shape = 1
out_shape = 1

# Training set
# Using linear regression to learn double function
x = np.random.randn(N, in_shape)
y = x * 2

linear_model = mikenet.Sequential(
  mikenet.Linear(in_shape, out_shape))

# Use stochastic gradient descent
#  and mean squared error for linear regression
mikenet.train(x, y, linear_model)

print()

# Testing set
loss = mikenet.MSE()
test_size = int(.3 * N)

test_x = x[-test_size:]
test_y = y[-test_size:]

preds = linear_model(test_x)

test_loss = loss(preds, test_y)

print(f'Test loss: {test_loss}')


print()
print('The model parameters should be close to weight = 2 and bias = 0')
print(f'Model weight: {linear_model.layers[0].params["w"]}')
print(f'Model bias: {linear_model.layers[0].params["b"]}')

