import torch

def nb_parameters(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    
    return n_params

def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def make_scheduler(name, optimizer):
    if name is None:
        return None
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)