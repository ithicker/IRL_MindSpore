
import mindspore.ops as ops
import mindspore 
from mindspore import nn
from losses import *
from model import MlpModel, LinearModel, ResNet, Bottleneck
from parameter import *
import copy
from mindspore import ParameterTuple
from mindspore.dataset import transforms,vision
from mindspore.train import Model,LossMonitor,MSE

class MyMAE(mindspore.train.Metric):
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        self._abs_error_sum = 0  
        self._samples_num = 0    

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        y = np.mean(y, axis=1)
        abs_error_sum = np.abs(y - y_pred)
        self._abs_error_sum += abs_error_sum.sum()

        self._samples_num += y.shape[0]

    def eval(self):
        return self._abs_error_sum / self._samples_num

def train_dataset_model(dataset_name, loss_type, model_type="mlp", metrics=None, print_show=False, out_dim=1):
    if metrics is None:
        metrics = ["mse"]
    if model_type == "mlp":
        net = MlpModel(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "linear":
        net = LinearModel(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "ResNet":
        net = ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim)
    else:
        print("invalid model")
        sys.exit()
    optimizer = mindspore.nn.Adam(net.trainable_params(), learning_rate=datasets["optim_rate"], beta1=0.9, beta2=0.999, eps=1e-08,
                           weight_decay=datasets["weight_decay"])

    best_model = {}
    max_loss = {}
    for metric in metrics:
        best_model[metric] = copy.deepcopy(net)
        if metric in ["mse", "mae", "gm"]:
            max_loss[metric] = MAX_INT
        elif metric in ["pearsonr"]:
            max_loss[metric] = 0
    if loss_type == "lm":
        loss_fn = L1mae()
    transform_train = transforms.Compose([
                vision.Resize((224, 224)),
                vision.RandomCrop(224, padding=16),
                vision.RandomHorizontalFlip(),
                vision.Normalize([.5, .5, .5], [.5, .5, .5], is_hwc=True),
                vision.ToTensor(),
            ])
    transform_val = transforms.Compose([
                vision.Resize((224, 224)),
                vision.Normalize([.5, .5, .5], [.5, .5, .5], is_hwc=True),
                vision.ToTensor(),
            ])       
    train_dataset = mindspore.dataset.GeneratorDataset(source=datasets[dataset_name]["train_dataset"],column_names=['data','label'],num_parallel_workers=datasets["num_work"])
    train_dataset = train_dataset.map(operations=transform_train, input_columns=["data"])
    train_dataset = train_dataset.batch(512,True )

    val_dataset = mindspore.dataset.GeneratorDataset(source=datasets[dataset_name]["verify_dataset"],column_names=['data','label'], num_parallel_workers=datasets["num_work"])
    val_dataset = val_dataset.map(operations=transform_val, input_columns=["data"])
    val_dataset = val_dataset.batch(512,True )    

    test_dataset = mindspore.dataset.GeneratorDataset(source=datasets[dataset_name]["test_dataset"],column_names=['data','label'],
                                        num_parallel_workers=datasets["num_work"])
    test_dataset = test_dataset.map(operations=transform_val, input_columns=["data"])
    test_dataset = test_dataset.batch(512,True )
    
    net_with_criterion = nn.WithLossCell(net, loss_fn)
    print("start training")
    model = Model(net, loss_fn, optimizer=optimizer, metrics = {"MAE": MyMAE()},)
    print("Init MAE Loss:",model.eval(test_dataset))
    model.train(epoch=1, train_dataset=train_dataset, callbacks=[LossMonitor()])
    print("Final MAE Loss:", model.eval(test_dataset))
    

    return model


