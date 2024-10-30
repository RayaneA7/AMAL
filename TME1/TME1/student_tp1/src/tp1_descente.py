import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context

p = 0.8

def exercise3_random():
    DATASET_SIZE = 1000
    DATA_TYPE = 'random-data'
    nx = 13
    ny = 3
    x = torch.randn(DATASET_SIZE, nx)
    y = torch.randn(DATASET_SIZE, ny)

    tests = [
        (int(p*DATASET_SIZE), 'GD'),
        (1, "S-GD"),
        (2, 'MINI-BATCH-GD-2'),
        (5, 'MINI-BATCH-GD-5'),
        (10, 'MINI-BATCH-GD-10'),
        (20, 'MINI-BATCH-GD-20'),
        (50, 'MINI-BATCH-GD-50'),
        (100, 'MINI-BATCH-GD-100'),
    ]
    return DATASET_SIZE,  DATA_TYPE, x, y, nx, ny, tests, 0.05

def exercise3_california():
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

    DATASET_SIZE = x.size(0)

    tests = [
        (int(p*DATASET_SIZE), 'GD'),
        (1, "S-GD"),
        (2, 'MINI-BATCH-GD-2'),
        (5, 'MINI-BATCH-GD-5'),
        (10, 'MINI-BATCH-GD-10'),
        (20, 'MINI-BATCH-GD-20'),
        (50, 'MINI-BATCH-GD-50'),
        (100, 'MINI-BATCH-GD-100'),
        (1000, 'MINI-BATCH-GD-1000'),
    ]

    return x.size(0), 'california-housing', x, y, x.size(1), y.size(1), tests, 0.001

# DATASET_SIZE,  DATA_TYPE, x, y, nx, ny, tests, epsilon = exercise3_random()
DATASET_SIZE,  DATA_TYPE, x, y, nx, ny, tests, epsilon = exercise3_california()


mse = MSE
linear = Linear

train_x = x[0:int(p*DATASET_SIZE)]
train_y = y[0:int(p*DATASET_SIZE)]
test_x = x[int(p*DATASET_SIZE):]
test_y = y[int(p*DATASET_SIZE):]

for batch_size, gd_name in tests:

    # Les paramètres du modèle à optimiser
    w = torch.randn(nx, ny,)
    b = torch.randn(ny,)

    writer = SummaryWriter(f"outputs/exercise3_{DATA_TYPE}_{gd_name}_DATASETSIZE_{DATASET_SIZE}")
    for n_iter in range(100):

        perm = list(range(len(train_x)))
        random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]

        train_cumloss = 0
        ii = 0

        ctx_linear = Context()
        ctx_mse = Context()
        
        for i in range(0, train_x.size(0), batch_size):

            train_x_i = train_x[i:i+batch_size]
            train_y_i = train_y[i:i+batch_size]

            pred = linear.forward(ctx_linear, train_x_i, w, b)

            loss_value = mse.forward(ctx_mse, pred, train_y_i)
            
            train_cumloss+=loss_value.detach().item()
            
            # Calcul du backward (grad_w, grad_b)
            yhat_grad, y_grad = mse.backward(ctx_mse, 1)
            grad_X, grad_W, grad_b = linear.backward(ctx_linear, yhat_grad)

            #Mise à jour des paramètres du modèle
            w.data -= grad_W * epsilon
            b.data -= grad_b * epsilon

            ii += (len(train_x_i) / batch_size )

        train_cumloss = train_cumloss/ii

        loss = train_cumloss

        pred = linear.forward(ctx_linear, test_x, w, b)
        test_loss_value = mse.forward(ctx_mse, pred, test_y)
        
        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar(f'exercise3_Loss_{DATA_TYPE}_{DATASET_SIZE}/train', loss, n_iter)
        writer.add_scalar(f'exercise3_Loss_{DATA_TYPE}_{DATASET_SIZE}/test', test_loss_value, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: train_loss {loss} / test_loss {test_loss_value} / ii {ii}")



