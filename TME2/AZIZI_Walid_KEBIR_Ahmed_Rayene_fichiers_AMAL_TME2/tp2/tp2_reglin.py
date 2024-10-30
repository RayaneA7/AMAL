# TME 1 - AMAL
# REALISE PAR:
# KEBIR Ahmed Rayene
# AZIZI Walid

from enum import Enum
import random
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(HighwayNetwork, self).__init__()
        self.num_layers = num_layers
        self.transform_gates = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.carry_gates = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            transform_gate = torch.sigmoid(self.transform_gates[i](x))  # Gate that controls how much to transform
            carry_gate = 1 - transform_gate  # Gate that controls how much to carry forward
            transformed = F.relu(self.linears[i](x))  # Transformed data
            x = transform_gate * transformed + carry_gate * x  # Combination of transformed and original data
        return x
    
def f(X,W,b):
    return torch.matmul(X, W) + torch.tile(b, (X.shape[0], 1))

def MSE(Yhat, Y):
    return torch.mean(torch.pow(Yhat-Y, 2))

class ModuleLineaire(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size=100):
        super(ModuleLineaire,self).__init__()
        self.lin_1 = torch.nn.Linear(nx, hidden_size)
        self.tanh_1 = torch.nn.Tanh()
        self.lin_2 = torch.nn.Linear(hidden_size, ny)
    def forward(self,x):
        return self.lin_2(self.tanh_1(self.lin_1(x)))

def get_module(case, nx, ny, hidden_size=100):
    if case==-1:
        W =  torch.nn.Parameter(torch.rand((nx, ny), dtype=torch.float32))
        b =  torch.nn.Parameter(torch.rand((1, nx), dtype=torch.float32))
        return [W, b]
    elif case==0:
        lin_1 = torch.nn.Linear(nx, hidden_size)
        tanh_1 = torch.nn.Tanh()
        lin_2 = torch.nn.Linear(hidden_size, ny)
        return [lin_1, tanh_1, lin_2]
    elif case==1:
        return torch.nn.Sequential(
            torch.nn.Linear(nx, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, ny),
        )
    elif case==2:
        return ModuleLineaire(nx, ny, hidden_size)
    elif case==3:
        return HighwayNetwork(nx, ny, 2)

def reglin(
    X,Y, 
    xtest, ytest, 
    writer, name, 
    batchsize=32, eps=1e-3, niter=100, 
    optimizer_cls=None,
    case=-1,
):
    name_2 = ""
    if case == -1:
        W, b = get_module(case, X.shape[1], Y.shape[1],)
        if optimizer_cls is not None:
            optim = optimizer_cls(params=[W,b],lr=eps)
            optim.zero_grad()
            name_2 = "by-hand"
    elif case==0:
        layers = get_module(case, X.shape[1], Y.shape[1],)
        params = []
        for l in layers:
            params.extend(l.parameters())
        optim = optimizer_cls(params=params, lr=eps)
        optim.zero_grad()
        name_2 = "many-layers"
    elif case in (1, 2):
        module = get_module(case, X.shape[1], Y.shape[1],)
        optim = optimizer_cls(params=module.parameters(),lr=eps)
        optim.zero_grad()
        layers = [module]
        name_2 = "one-module-sequential" if case==1 else "one-module-class"
    elif case in (3,):
        module = get_module(case, X.shape[1], Y.shape[1],)
        optim = optimizer_cls(params=module.parameters(),lr=eps)
        optim.zero_grad()
        layers = [module]
        name_2 = "highway-neural-networks"

    train_loss_evol, test_loss_evol = [], []

    mse_loss_torch = torch.nn.MSELoss()

    for iter in range(niter):
        
        train_cumloss = 0
        ii = 0

        for i in range(0, X.shape[0], batchsize):

            xtrain = X[i:i+batchsize]
            ytrain = Y[i:i+batchsize]

            if optimizer_cls is None and case == -1:

                loss_value = MSE(f(xtrain, W, b), ytrain)
                train_cumloss+=loss_value.detach().item()
                loss_value.backward()

                W.data -= W.grad * eps
                b.data -= b.grad * eps
                W.grad.zero_()
                b.grad.zero_()
            
            elif case==-1:
                output = f(xtrain, W, b)
                loss_value = mse_loss_torch(output, ytrain)
                train_cumloss+=loss_value.detach().item()
                loss_value.backward()
                optim.step()
                optim.zero_grad()

            else:
                output = xtrain
                for l in layers:
                    output = l.forward(output)
                loss_value = mse_loss_torch(output, ytrain)
                train_cumloss+=loss_value.detach().item()
                loss_value.backward()
                optim.step()
                optim.zero_grad()

            ii += 1
        train_cumloss = train_cumloss/ii

        with torch.no_grad():
            if case == -1:
                pred = f(xtest, W, b)
                test_loss_value = MSE(pred, ytest).detach().item()
            else:
                output = xtest
                for l in layers:
                    output = l.forward(output)
                    loss_value = mse_loss_torch(output, ytest)
                    test_loss_value = loss_value.detach().item()


        print(f"{name_2=} {iter=}  \t {train_cumloss=} \t {test_loss_value=}")

        train_loss_evol.append(train_cumloss)
        test_loss_evol.append(test_loss_value)

        writer.add_scalar(name+'/Loss/train',train_cumloss, iter)
        writer.add_scalar(name+'/Loss/test',test_loss_value, iter)
        
    return train_loss_evol, test_loss_evol

def exercise_1():

    DATASET_SIZE = 1000

    p = 0.8
    nx = 5
    train_x = torch.rand((int(DATASET_SIZE*p), nx))
    test_x = torch.rand((int(DATASET_SIZE*(1-p)), nx))
    ny = 1
    train_y = torch.rand((int(DATASET_SIZE*p), ny))
    test_y = torch.rand((int(DATASET_SIZE*(1-p)), ny))

    return DATASET_SIZE, train_x, train_y, test_x, test_y

def exercise_2():

    from sklearn.datasets import fetch_california_housing
    california_housing = fetch_california_housing(as_frame=True)
    x = torch.tensor(california_housing.data.to_numpy(), dtype=torch.float32)
    y = torch.tensor(california_housing.target.to_numpy(),dtype=torch.float32).unsqueeze(1)

    perm = list(range(len(x)))
    random.shuffle(perm)

    x = x[perm]
    y = y[perm]

    mean, std, var = (
        torch.mean(x, dim=0), 
        torch.std(x, dim=0), 
        torch.var(x, dim=0) 
    )
    x = (x - mean)
    x = x / std

    p = 0.8

    DATASET_SIZE = x.shape[0]

    train_x = x[0:int(p*DATASET_SIZE)]
    train_y = y[0:int(p*DATASET_SIZE)]
    test_x = x[int(p*DATASET_SIZE):]
    test_y = y[int(p*DATASET_SIZE):]

    return DATASET_SIZE, train_x, train_y, test_x, test_y

if __name__ == '__main__':

    (DATASET_SIZE, train_x, train_y, test_x, test_y), exercise = (
        exercise_1(), 
        'exercise1_random_data',
    )

    # (DATASET_SIZE, train_x, train_y, test_x, test_y), exercise = (
    #     exercise_2(),
    #     'exercise2_california_housing_data',
    # )

    N_ITER = 100
    BATCH_SIZE = 32
    eps = 0.001

    # Hand stochastic descent
    writer = SummaryWriter('outputs/'+exercise+'_Hand')
    train_loss_evol, test_loss_evol = reglin(train_x, train_y, test_x, test_y, niter=N_ITER, batchsize=BATCH_SIZE, eps=eps, writer=writer, name=exercise, optimizer_cls=None)
    # Adam descent
    writer = SummaryWriter('outputs/'+exercise+'_Adam')
    train_loss_evol, test_loss_evol = reglin(train_x, train_y, test_x, test_y, niter=N_ITER, batchsize=BATCH_SIZE, eps=eps, writer=writer, name=exercise, optimizer_cls=torch.optim.Adam)
    # SGD descent
    writer = SummaryWriter('outputs/'+exercise+'_SGD')
    train_loss_evol, test_loss_evol = reglin(train_x, train_y, test_x, test_y, niter=N_ITER, batchsize=BATCH_SIZE, eps=eps, writer=writer, name=exercise, optimizer_cls=torch.optim.SGD)

    exercise = 'exercise2'
    # Using many layers
    writer = SummaryWriter('outputs/'+exercise+'layers_Adam')
    train_loss_evol, test_loss_evol = reglin(
        train_x, train_y, 
        test_x, test_y, 
        niter=N_ITER, 
        batchsize=BATCH_SIZE, 
        eps=eps, 
        writer=writer, 
        name=exercise, 
        case=0,  # layers
        optimizer_cls=torch.optim.Adam
    )
    # Using the sequential module
    writer = SummaryWriter('outputs/'+exercise+'sequential_Adam')
    train_loss_evol, test_loss_evol = reglin(
        train_x, train_y, 
        test_x, test_y, 
        niter=N_ITER, 
        batchsize=BATCH_SIZE, 
        eps=eps, 
        writer=writer, 
        name=exercise, 
        case=1,  # sequential
        optimizer_cls=torch.optim.Adam
    )
    # Using the module class
    writer = SummaryWriter('outputs/'+exercise+'one-class-module_Adam')
    train_loss_evol, test_loss_evol = reglin(
        train_x, train_y, 
        test_x, test_y, 
        niter=N_ITER, 
        batchsize=BATCH_SIZE, 
        eps=eps, 
        writer=writer, 
        name=exercise, 
        case=2,  # one class module
        optimizer_cls=torch.optim.Adam
    )
    # Using the highway network class
    writer = SummaryWriter('outputs/'+exercise+'highway-network_Adam')
    train_loss_evol, test_loss_evol = reglin(
        train_x, train_y, 
        test_x, test_y, 
        niter=N_ITER, 
        batchsize=BATCH_SIZE, 
        eps=eps, 
        writer=writer, 
        name=exercise, 
        case=3,  # one class module
        optimizer_cls=torch.optim.Adam
    )
    writer.close()
    
