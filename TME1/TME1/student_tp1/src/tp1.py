
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        return ((y - yhat)**2).sum() / y.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        yhat_grad = grad_output * (2 * (yhat - y)) / y.size(0)
        y_grad = grad_output * (2 * (-1) * (yhat - y) / y.size(0)) 
        return yhat_grad, y_grad
 

class Linear(Function):
    """Implementation of Linear function with autograd support"""
    @staticmethod
    def forward(ctx, X, W, b):
        # Save tensors for backward
        ctx.save_for_backward(X, W, b)
        # Compute linear transformation
        return torch.matmul(X, W) + b

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        X, W, b = ctx.saved_tensors
        # Gradients with respect to X, W, and b
        grad_X = torch.matmul(grad_output, W.t()) 
        grad_W = torch.matmul(X.t(),grad_output) 
        grad_b = torch.sum(grad_output, dim=0)
        return grad_X, grad_W, grad_b

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

