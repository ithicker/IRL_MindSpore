import mindspore.ops as ops
import mindspore.nn as nn
import warnings
from parameter import *
from scipy.stats import gmean, pearsonr
import numpy as np
warnings.filterwarnings("ignore")
from mindspore.nn.loss.loss import LossBase


# ----------------------------------------------------------------------------------------------------------------------
def l1_mae(pre, interval):
    """LM"""
    loss_1 = ops.unsqueeze(interval[:, 0], dim=1) - pre
    loss_2 = pre - ops.unsqueeze(interval[:, 1], dim=1)
    loss = ops.relu(loss_1) + ops.relu(loss_2)
    return ops.mean(loss)
class L1mae(LossBase):
    def __init__(self):
        """There are two inputs, the forward network backbone and the loss function"""
        super(L1mae, self).__init__()
    def construct(self, pre, interval):
        loss_1 = ops.unsqueeze(interval[:, 0], dim=1) - pre
        loss_2 = pre - ops.unsqueeze(interval[:, 1], dim=1)
        loss = ops.relu(loss_1) + ops.relu(loss_2)
        return ops.mean(loss)

# ----------------------------------------------------------------------------------------------------------------------