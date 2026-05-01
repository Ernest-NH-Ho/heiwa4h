import torch

class EntropyLoss:
    eps = 1
    
    @staticmethod
    def calculate(tensor:torch.tensor, n:torch.tensor):
        return (tensor * torch.log(tensor + EntropyLoss.eps)).sum() / torch.log(n.sum())

class KurtosisLoss:
    eps = 1e-7
    
    @staticmethod
    def calculate(tensor:torch.tensor, n:torch.tensor):
        n_sum = n.sum()
        avg = tensor.sum() / n_sum
        num = ((tensor - avg)**4).sum() / n_sum
        den = (((tensor - avg)**2 + KurtosisLoss.eps).sum() / n_sum)**2
        return -num / den