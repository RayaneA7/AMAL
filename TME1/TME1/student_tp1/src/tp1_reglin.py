import random
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def f(X,W,b):
    return torch.matmul(X, W) + torch.tile(b, (X.shape[0], 1))

def MSE(Yhat, Y):
    return torch.mean(torch.pow(Yhat-Y, 2))

def reglin(X,Y, xtest, ytest, writer, name, batchsize=32, eps=1e-3, niter=100):
    W = torch.rand((X.shape[1], Y.shape[1]), requires_grad=True, dtype=torch.float32)
    W.retain_grad()
    b = torch.rand((1, X.shape[1]), requires_grad=True, dtype=torch.float32)
    b.retain_grad()

    train_loss_evol, test_loss_evol = [], []

    for iter in range(niter):

        train_cumloss = 0
        ii = 0
        for i in range(0, X.shape[0], batchsize):

            xtrain = X[i:i+batchsize]
            ytrain = Y[i:i+batchsize]

            pred = f(xtrain, W, b)
            loss_value = MSE(pred, ytrain)
            
            train_cumloss+=loss_value.detach().item()
            loss_value.backward()

            W.data -= W.grad * eps
            b.data -= b.grad * eps

            W.grad.zero_()
            b.grad.zero_()

            ii += 1
        train_cumloss = train_cumloss/ii

        with torch.no_grad():
            pred = f(xtest, W, b)
            test_loss_value = MSE(pred, ytest).detach().item()

        print(f"{iter=}  \t {train_cumloss=} \t {test_loss_value=}")

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

    # (DATASET_SIZE, train_x, train_y, test_x, test_y), exercise = (
    #     exercise_1(), 
    #     'exercise1_random_data',
    # )

    (DATASET_SIZE, train_x, train_y, test_x, test_y), exercise = (
        exercise_2(),
        'exercise2_california_housing_data',
    )

    N_ITER = 100
    BATCH_SIZE = 32
    eps = 0.001

    writer = SummaryWriter('outputs/'+exercise)

    train_loss_evol, test_loss_evol = reglin(train_x, train_y, test_x, test_y, niter=N_ITER, batchsize=BATCH_SIZE, eps=eps, writer=writer, name=exercise)

    df = pd.DataFrame.from_dict({
        'train' : train_loss_evol,
        'test' : test_loss_evol,
        'iter' : list(range(len(train_loss_evol))),
    })
    df = df.melt(id_vars=['iter'], value_vars=['train','test'], var_name='loss_type',value_name='loss' )
    fig = plt.figure()
    sns.lineplot(df, x='iter', y='loss', hue='loss_type')
    fig.savefig(f'outputs/{exercise}/{exercise}.png')

    writer.close()
    
