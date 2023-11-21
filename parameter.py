import sys
from data.data_set import AgeDB
from data.data_processing import AgeDB_data_processing
import os
import numpy as np
import random
import pandas as pd
import mindspore
def up_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)


MAX_INT = sys.maxsize
OPTIM_RATE = [0.001, 0.01]

epoch = 100
optim_rate = OPTIM_RATE[0]

seed = 2023
up_seed(2023)

datasets = {
    "agedb": {"fun_data_processing": AgeDB_data_processing, "class_dataset": AgeDB},
    "optim_rate": optim_rate, "epoch": epoch,
    "num_work": 1, "pin_memory": False, "non_blocking": False, "batch_size": 256, "weight_decay": 0,
    "delta": 5, "lamda": 10,
    "max_interval": 40, "yl": 0, "yr": 200
}


def load_data(data_name, max_interval, time=1, ratio=1, left=-1, right=7):
    df = pd.read_csv(
    	"data/dataset/" + data_name + "_split/" + data_name + "_" + str(max_interval) + "_" + str(time) + ".csv",
    	sep=",")
    train_data = df.loc[df['split'] == "train"]
    if ratio != 1:
    	train_data = slicing(train_data, ratio)
    verify_data = df.loc[df['split'] == "verify"]
    test_data = df.loc[df['split'] == "test"]
    
    train_interval = train_data[["yl", "yr"]].values
    train_interval = np.float32(train_interval)
    train_number = mindspore.Tensor(9892)
    datasets[data_name]["train_dataset"] = AgeDB(df=train_data,
    																			data_dir="data/dataset",
    																			img_size=224, number=train_number,
    																			interval=train_interval)
    
    verify_interval = verify_data[["yl", "yr"]].values
    verify_interval = np.float32(verify_interval)
    verify_number = mindspore.tensor(2000)
    datasets[data_name]["verify_dataset"] = AgeDB(df=verify_data,
    																			 data_dir="data/dataset",
    																			 img_size=224, number=verify_number,
    																			 interval=verify_interval)
    
    test_interval = test_data[["yl", "yr"]].values
    test_interval = np.float32(test_interval)
    test_number = mindspore.tensor(2000)
    datasets[data_name]["test_dataset"] = AgeDB(df=test_data,
    																		   data_dir="data/dataset",
    																		   img_size=224, number=test_number,
    																		   interval=test_interval)