from train import *
import sys
from data import *
import os
import numpy as np
import random 
import  pandas as pd
import mindspore
def up_seed(rand_seed):
    """"update seed"""
    np.random.seed(rand_seed)
    random.seed(rand_seed)

MAX_INT = sys.maxsize
OPTIM_RATE = [0.001, 0.01]

epoch = 100
optim_rate = OPTIM_RATE[0]

seed = 2023
up_seed(2023)

datasets["non_blocking"] = False
datasets["pin_memory"] = False

dataset_name = "agedb"
model_type = "ResNet"
loss_type = "lm"
max_interval = 40

datasets["epoch"] = 10 #100
datasets["optim_rate"] = 0.001
datasets["batch_size"] = 256
out_dim = 1
if loss_type in ["CRM", "RANN", "SINN", "IN"]:
    out_dim = 2
metrics = ["mse"]

load_data(dataset_name, max_interval)
best_model = train_dataset_model(dataset_name, loss_type, model_type=model_type,
                                 metrics=metrics,
                                 print_show=True, out_dim=out_dim)
