import copy, os, socket, sys, time
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import data, fl, nn, wandb
from libs.distributed import *

class FedArgs():
    def __init__(self):
        self.name = "client-x"
        self.num_clients = 1
        self.epochs = 51
        self.local_rounds = 1
        self.client_batch_size = 32
        self.test_batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay = None
        self.cuda = False
        self.seed = 1
        self.topic = "VJH_020_2"
        self.broker_ip = '172.16.26.40:9092'
        self.schema_ip = 'http://172.16.26.40:8081'
        self.wait_to_consume = 10
        self.dataset = "lemon"
        self.model = nn.LemonNet()
        self.train_func = fl.train_model
        self.eval_func = fl.evaluate
        
        
fedargs = FedArgs()

project = 'fl-kafka-client'
def setup_project(_name):
    name = 'VJH_020_1-' + _name
    fedargs.wb = wandb.init(name, project)
    host = socket.gethostname()
    fedargs.clients = [host + ": " + _name]
    fedargs.dt = Distributed(fedargs.clients, fedargs.broker_ip, fedargs.schema_ip, fedargs.wait_to_consume)

use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

# Load Data to clients
train_loader, test_loader = data.load_dataset(fedargs.dataset, fedargs.client_batch_size, fedargs.test_batch_size)

def process(epoch = 1):
    client = fedargs.clients[0]
    # Consume Models
    client_model_updates = fedargs.dt.consume_model(client, fedargs.topic, fedargs.model, epoch)
    if client in client_model_updates:
        client_model_updates.pop(client)
    print("Epoch: {}, Processing Client {}, Received {} Updates From {}".format(epoch, client, 
                                                                                len(client_model_updates), 
                                                                                list(client_model_updates.keys())))

    #if len(client_model_updates) != 0:
    #    model = fl.federated_avg(client_model_updates)
    
    # Train
    model_update, fedargs.model, loss = fedargs.train_func(fedargs.model, train_loader, 
                                                   fedargs.learning_rate,
                                                   fedargs.weight_decay,
                                                   fedargs.local_rounds, device)
    
    # Publish Model
    epoch = epoch + 1
    fedargs.dt.produce_model(client, fedargs.topic, fedargs.model, epoch)

    # Test, Plot and Log
    test_output = fedargs.eval_func(fedargs.model, test_loader, device)
    print("Epoch: {}, Accuracy: {}, Test Loss: {}".format(epoch, test_output["accuracy"], test_output["test_loss"]))
    fedargs.wb.log({client: {"epoch": epoch, "time": time.time(), "acc": test_output["accuracy"], "loss": test_output["test_loss"]}})

    return test_output["accuracy"], test_output["test_loss"] 

# Federated Training
#for epoch in tqdm(range(fedargs.epochs)):
#    print("Federated Training Epoch {} of {}".format(epoch, fedargs.epochs))

from torchvision import transforms
from torchvision.transforms.functional import crop
def crop800(image):
    return crop(image, 300, 300, 500, 500)

def health_meter(image):
    data_transforms=transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                        transforms.Lambda(crop800), 
                                        transforms.Resize((50,50)), 
                                        transforms.ToTensor()])
    img = data_transforms(image)
    img = img.unsqueeze(0)
    out = fedargs.model(img)
    _, preds = torch.max(out, dim=1)
    health_index = preds.item()
    print("Health Index:", health_index)
    return health_index