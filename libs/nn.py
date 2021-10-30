import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.federated.model_serialization import deserialize_model_params

class ModelMNIST(nn.Module):
    def __init__(self):
        super(ModelMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def getModel(pb, _model):
    ser_params = pb.SerializeToString()
    params = deserialize_model_params(ser_params)
    
    _model_dict = _model.state_dict()
    
    for index, key in enumerate(_model_dict):
        _model_dict[key] = params[index]
    _model.load_state_dict(_model_dict)
    
    return _model

class LemonNet(nn.Module):
    def __init__(self): 
        super(LemonNet,self).__init__()
        self.cnv1 = nn.Conv2d(1, 5, 7)
        self.cnv2 = nn.Conv2d(5, 10, 7)
        self.ful1 = nn.Linear(14440, 100)
        self.ful2 = nn.Linear(100, 50)
        self.ful3 = nn.Linear(50, 10)
        
    def forward(self,x):
        x = F.elu(self.cnv1(x))
        x = F.elu(self.cnv2(x))
        x = x.view(-1,14440)
        x = F.elu(self.ful1(x))
        x = F.elu(self.ful2(x))
        x = self.ful3(x)       
        return x