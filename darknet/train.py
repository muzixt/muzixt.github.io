import glob
import os
import random
import shutil
import sys
import time
import datetime
from collections import Counter
from typing import List
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from model import darknet53
from net2 import *
from progress_bar import ProgressBar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(777)


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


def train_step(model, features, labels, loss_func, optimizer):
    model.train()
    optimizer.zero_grad()

    features = features.to(device)
    labels = labels.to(device)

    # features, labels = Variable(features), Variable(labels)
    # forward
    predictions = model(features)

    predictions = torch.as_tensor(predictions, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.float)

    # print(predictions.dtype)
    # print(labels.dtype)
    # print(labels.squeeze().dtype)
    loss = loss_func(predictions, labels.squeeze().long())
    global weight_decay, reg_loss
    if weight_decay > 0:
        loss = loss + reg_loss(model)

    # loss=loss_func(predictions,labels)
    # evaluate metrics
    prec = accuracy(predictions.data, labels.data, topk=(1,))

    train_metrics = {"train_loss": loss.item()}
    for i, p in enumerate(prec):
        train_metrics["train_acc_top" + str(i + 1)] = p.item()
    # backward
    loss.backward()

    # update parameters

    optimizer.step()

    return train_metrics


@torch.no_grad()
def evaluate_step(model, features, labels, loss_func):
    model.eval()

    features = features.to(device)
    labels = labels.to(device)

    # features, labels = Variable(features), Variable(labels)

    with torch.no_grad():
        predictions = model(features)

        predictions = torch.as_tensor(predictions, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)

        loss = loss_func(predictions, labels.squeeze().long())

    prec = accuracy(predictions.data, labels.data, topk=(1,))

    val_metrics = {"val_loss": loss.item()}
    for i, p in enumerate(prec):
        val_metrics["val_acc_top" + str(i + 1)] = p.item()

    return val_metrics


# batch val
@torch.no_grad()
def evaluate(model, dl_val, loss_func):
    model.eval()
    val_metrics_list = {}
    evaluate_log = {}
    # -----------------bar-----------------
    evaluate_iters = len(dl_val)
    evaluate_progress = ProgressBar("Evaluate", epoch=1, total_epoch=1, iters=evaluate_iters, width=10)
    # ---------------------------

    for it, (features, labels) in enumerate(dl_val):
        val_metrics = evaluate_step(model, features, labels, loss_func)

        for name, metric in val_metrics.items():
            val_metrics_list[name] = val_metrics_list.get(name, []) + [metric]

        for name, metric_lst in val_metrics_list.items():
            evaluate_log.update({name: np.mean(metric_lst)})

        #     ----------------------------
        evaluate_progress.update((it + 1), message=evaluate_log)
        evaluate_progress()

    return {name: np.mean(metric_list) for name, metric_list in val_metrics_list.items()}


@torch.no_grad()
def predict(model, dl):
    model.eval()
    result = torch.cat([model(t[0].to(device)) for t in dl])
    target = torch.cat([t[1] for t in dl])
    return result.data, target.data


# print(accuracy(torch.Tensor(torch.range(1, 25).view(5, 5)), torch.Tensor(torch.range(1, 5).view(5))))
def save_log_to_txt(dic: dict, path="./log", start_time=""):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = start_time + "_checkpoint.txt"
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        text = "epoch   train_loss   train_acc_top1   val_loss   val_acc_top1 \n"
    else:
        text = ""
    for i, v in dic.items():
        text += str(round(v, 5)) + "\t"
    text += "\n"
    with open(filepath, mode="a+") as f:
        f.write(text)


def save_checkpoint(state, is_best, filename='model.pth', start_time=""):
    model_dir = "./log/model"
    os.makedirs(model_dir, exist_ok=True)

    tmp_dirs = [f for f in glob.glob(os.path.join(model_dir, start_time + '*')) if os.path.isfile(f)]
    name = start_time + "_epoch" + str(len(tmp_dirs) + 1) + "_" + filename

    model_path = os.path.join(model_dir, name)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(model_dir, "Best_" + start_time + '_model.pth'))


def train(model, train_loader, val_loader, start_epoch=1, end_epoch=200, loss_func=None, optimizer=None,
          scheduler=None):
    print("Start Training ...")

    best_acc1 = 0

    train_iters = len(train_loader)
    val_iters = len(val_loader)
    train_progress = ProgressBar("Train", total_epoch=end_epoch, iters=train_iters, width=30)
    val_progress = ProgressBar("Val", total_epoch=end_epoch, iters=val_iters, width=10)

    epoch_log = {}
    for epoch in range(start_epoch, end_epoch + 1):

        log_dic = {"epoch": epoch}
        # -----------------------------train-------------------------------
        train_metrics_sum, it = {}, 0
        for features, labels in train_loader:
            it += 1
            train_metrics = train_step(model, features, labels, loss_func, optimizer)

            for name, metric in train_metrics.items():
                train_metrics_sum[name] = train_metrics_sum.get(name, 0.0) + metric
            for name, metric_sum in train_metrics_sum.items():
                epoch_log.update({name: metric_sum / it})
            else:
                log_dic.update(epoch_log)
            # --------------bar-------------
            train_progress.update(it, epoch, epoch_log)
            train_progress()
            epoch_log.clear()

        # ---------------------------------val-----------------------------------------

        val_metrics_sum, it = {}, 0
        for features, labels in val_loader:
            it += 1
            val_metrics = evaluate_step(model, features, labels, loss_func)

            for name, metric in val_metrics.items():
                val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric

            for name, metric_sum in val_metrics_sum.items():
                epoch_log.update({name: metric_sum / it})
            else:
                log_dic.update(epoch_log)
            # --------------bar-------------
            val_progress.update(it, epoch, epoch_log)
            val_progress()
            epoch_log.clear()
        # -------update lr---------
        if scheduler:
            metric = log_dic['val_acc_top1'] if scheduler.mode == "max" else log_dic['val_loss']
            scheduler.step(metric)
        # -------------save log -----------------
        save_log_to_txt(log_dic, start_time=start_time)
        # ----------save checkpoint----------------------------
        val_acc = log_dic['val_acc_top1']
        if is_best := val_acc > best_acc1:
            best_acc1 = val_acc
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, filename='model.pth', start_time=start_time)
        # ---------------------------------------------------------
        log_dic.clear()


class hisEqulColorTransform:

    def __init__(self, num=2):
        self.num = num

    def __call__(self, x):
        for _ in range(self.num):
            x = self.hisEqulColor(x)
        return x

    # 自适应直方图均衡化
    def hisEqulColor(self, img):
        # 转cv格式
        ycrcb = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2YCR_CB)
        channels = cv.split(ycrcb)
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(7, 7))
        clahe.apply(channels[0], channels[0])
        cv.merge(channels, ycrcb)
        # 转PIL格式
        img = Image.fromarray(cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2RGB))
        return img


def main():
    start_epoch = 1
    end_epoch = 200
    freeze_eopch = 0
    batch_size = 128
    num_workers = 8
    model_path = "./model/Best1.pth"

    global weight_decay, reg_loss
    weight_decay = 0.01  # 正则化参数

    global start_time
    timestamp = time.time()
    datetime_struct = datetime.datetime.fromtimestamp(timestamp)
    start_time = datetime_struct.strftime('%Y-%m-%d-%H-%M-%S-%f')

    transform_train = transforms.Compose([
        # hisEqulColorTransform(),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.Resize((224, 224)),
        # transforms.RandomVerticalFlip(p=0.7),
        # transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #
    ds_train = datasets.ImageFolder("./data/train/",
                                    transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())

    ds_valid = datasets.ImageFolder("./data/val/",
                                    transform=transform_val, target_transform=lambda t: torch.tensor([t]).float())

    ds_test = datasets.ImageFolder("./data/test/",
                                   transform=transform_test, target_transform=lambda t: torch.tensor([t]).float())

    trainLoader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                             prefetch_factor=4, persistent_workers=True)
    validLoader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                             prefetch_factor=4, persistent_workers=True)
    testLoader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # ------ load model ------------
    model = resnet34(num_classes=5).to(device)

    # ----------is  Load pretrained  --------
    print("Load pretrained ...")
    state = torch.load(model_path)

    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in state.items() if (k in model_dict and 'fc' not in k)}
    # # 更新权重
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    model.load_state_dict(state['state_dict'], strict=False)

    # optimizer.load_state_dict(state['optimizer'])

    # -----------loss  and scheduler and Regularization---------------
    Loss = nn.CrossEntropyLoss()

    if weight_decay > 0:
        reg_loss = Regularization(model, weight_decay, p=2).to(device)
    else:
        print("no regularization")

    # ----------start training-------------
    if freeze_eopch:
        print("freeze train...")
        # freeze
        for p in model.named_parameters():
            # print(p[1].requires_grad)
            if "fc" not in p[0]:
                p[1].requires_grad = False
                # print(p[0], p[1].requires_grad)
        optimizer = torch.optim.Adam(
            [{'params': [param for name, param in model.named_parameters() if 'fc' not in name]}], lr=0.001)

        # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=7)
        train(model, trainLoader, validLoader, start_epoch, freeze_eopch, Loss, optimizer)
        # un freeze
        print("un freeze train...")
        for p in model.named_parameters():
            # print(p[1].requires_grad)
            if "fc" not in p[0]:
                p[1].requires_grad = True

        optimizer.add_param_group({'params': [param for name, param in model.named_parameters() if 'fc' in name]})

        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)
        train(model, trainLoader, validLoader, freeze_eopch + 1, end_epoch, Loss, optimizer, scheduler)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)  # min
        train(model, trainLoader, validLoader, start_epoch, end_epoch, Loss, optimizer, scheduler)

    # results = evaluate(model, testLoader, loss_func=Loss)


if __name__ == '__main__':
    main()
