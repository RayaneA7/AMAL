import torch
from tp1 import mse, linear

# Test du gradient de MSE
yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
mse_test = torch.autograd.gradcheck(mse, (yhat, y))
print('MSE Gradcheck:', mse_test)

# Test du gradient de Linear (sur le même modèle que MSE)
batch, nx = 100, 5
nh = 10
X = torch.randn(batch,nx, requires_grad=True, dtype=torch.float64)
W = torch.randn(nx, nh, requires_grad=True, dtype=torch.float64)
b = torch.randn(nh, requires_grad=True, dtype=torch.float64)
linear_test = torch.autograd.gradcheck(linear, (X, W, b))
print('Linear Gradcheck:', linear_test)
