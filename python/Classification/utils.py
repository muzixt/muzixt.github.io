import random
import os
import numpy as np
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, CohenKappa
from typing import List
import sys
import re
import time
import torch
from rich.progress import track
import time
from time import sleep


def random_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch
    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
                y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def get_metrics(num_classes: int, task="multiclass"):
    metrics = MetricCollection({
        'acc': Accuracy(task=task, num_classes=num_classes, top_k=1,average="micro"),
        'pre': Precision(task=task, num_classes=num_classes, average='micro'),
        'rec': Recall(task=task, num_classes=num_classes, average='micro'),
        'f1': F1Score(task=task, num_classes=num_classes, average='micro'),
        'kappa': CohenKappa(task=task, num_classes=num_classes)
    })
    return metrics


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, iters=None, current=None, width=30, symbol="\u2588",
                 output=sys.stdout):  # stdout
        assert len(symbol) == 1

        self.mode = mode
        self.iters = iters
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.txt = ""
        self.args = {}
        self.pre_time = 0.0
        self.cur_time = 0.0
        self.total_time = 0.0
        self.started_time = 0.0

    def update(self, current, epoch=None, message: dict = {}):
        self.current = current
        if epoch:
            self.epoch = epoch
        txt = ""
        for k, v in message.items():
            txt += f'{k}:{v:.3f} '
        self.txt = txt
        self.__call__()

    def sec2time(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        pattern = r'%02d:%02d:%02d'
        return pattern % (h, m, s)

    def __call__(self):
        percent = self.current / float(self.iters)
        size = int(self.width * percent)
        bar = "\u2502" + self.symbol * size + " " * (self.width - size) + "\u2502"

        self.pre_time = time.time() if self.pre_time == 0 else self.pre_time

        self.cur_time = time.time()
        spend_time = self.cur_time - self.pre_time
        self.pre_time = self.cur_time

        self.started_time += spend_time
        self.total_time = self.started_time + spend_time * (self.iters - self.current)

        self.args.update({
            "mode": self.mode,
            "iters": self.iters,
            "bar": bar,
            "current": self.current,
            "percent": percent,
            "txt": self.txt,
            "epoch": self.epoch,
            "epochs": self.total_epoch,
            "time": f"{spend_time:.4f}",
            "started_time": self.sec2time(self.started_time),
            "total_time": self.sec2time(self.total_time)
        })
        message = "\033[1;31m{mode} Epoch: {epoch}/{epochs}\033[0m \033[1;33m {bar} \033[0m  \033[1;32m[ {txt} ]\033[0m \033[1;36m[ {current}/{iters} | {time} sec/it | {started_time}/{total_time} | {percent:.2%} ]\033[0m".format(
            **self.args)
        if self.current == self.iters:
            print("\r", message, file=self.output)
            self.started_time = 0.0
            self.total_time = 0.0
            self.pre_time = 0.0
            # print(self.current,self.total)
        else:
            print("\r" + message, file=self.output, end="")


if __name__ == "__main__":

    progress = ProgressBar("Train", total_epoch=10, iters=100, )
    progress1 = ProgressBar("val", total_epoch=10, iters=100, width=10)
    for i in range(1, 10):
        for x in range(100):
            progress.update(x + 1, i, message={"mess": 1.00})
            # progress()
            sleep(0.01)
            progress1.update(x + 1, i, message={"mess": 1.00})
            # progress1()

    # progress.done()
