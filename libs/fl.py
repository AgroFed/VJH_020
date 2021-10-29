import copy
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import optim

def client_update(_model, data_loader, learning_rate, decay, epochs, device):
    model = copy.deepcopy(_model)
    loss = {}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=decay)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _loss = F.cross_entropy(output, target)
            _loss.backward()
            optimizer.step()

        loss["Epoch " + str(epoch + 1)] = _loss.item()
    return model, loss

def evaluate(model, test_loader, device, flip_labels = None):
    model.eval()
    test_output = {
        "test_loss": 0,
        "correct": 0,
        "accuracy": 0
    }

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_output["test_loss"] += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            if flip_labels is not None and len(flip_labels) > 0:
                audit_attack(target, pred, flip_labels, test_output["attack"])
            test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

    test_output["test_loss"] /= len(test_loader.dataset)
    test_output["accuracy"] = (test_output["correct"] / len(test_loader.dataset)) * 100

    return test_output

def federated_avg(models: Dict[Any, torch.nn.Module],
                  base_model: torch.nn.Module = None) -> torch.nn.Module:
    if len(models) > 1:
        model_list = list(models.values())
        model = reduce(add_model, model_list)
        model = scale_model(model, 1.0 / len(models))
        if base_model is not None:
            model = sub_model(base_model, model)
        return model
    else:
        model = copy.deepcopy(list(models.values())[0])
    return model

def sub_model(model1, model2):
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] - params2[name1]
    model = copy.deepcopy(model1)
    model.load_state_dict(params1, strict=False)
    return model

def train_model(_model, train_loader, lr, wd, r, device):
    model, loss = client_update(_model, train_loader, lr, wd, r, device)
    model_update = sub_model(_model, model)
    return model_update, model, loss