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
import csv
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from network import darknet53, Resnet
from script.dataset import datasets
from utils import ProgressBar, Regularization, get_metrics, random_seeds
from Config import Config

random_seeds(777)


def save_log_to_csv(log: dict, path="./log", start_time=""):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = start_time + "_checkpoint.csv"
    filepath = os.path.join(path, filename)
    flag = False
    if not os.path.exists(filepath):
        flag = True
        header = ['epoch', 'loss', 'acc', "pre", "rec", "f1", "kappa", 'loss', 'acc', "pre", "rec", "f1", "kappa"]
    item = [log['epoch']]
    for t in "train", "val":
        for k in 'loss', 'acc', "pre", "rec", "f1", "kappa":
            item.append(log[t][k].item())
    with open(filepath, 'a+', encoding='utf-8', newline="") as file_obj:
        writer = csv.writer(file_obj)
        if flag:
            writer.writerow(header)
        writer.writerow(item)


def save_checkpoint(state, model_dir, is_best, start_time=""):
    model_dir = os.path.join(model_dir, start_time)
    os.makedirs(model_dir, exist_ok=True)
    tmp_dirs = [f for f in glob.glob(os.path.join(model_dir, 'Epoch_*')) if os.path.isfile(f)]
    name = "Epoch_" + str(len(tmp_dirs) + 1) + ".pt"
    model_path = os.path.join(model_dir, name)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(model_dir, "Best_" + 'model.pt'))


def train_step(model, features, labels, loss_func, train_metrics, optimizer, regularization=None):
    model.train()
    optimizer.zero_grad()
    features = features.to(device)
    labels = labels.to(device)
    predictions = model(features)
    loss = loss_func(predictions, labels)
    if regularization:
        loss = loss + regularization(model)
    train_metrics.update(predictions, labels)
    metrics = train_metrics.compute()
    train_loss = {"loss": loss, **metrics}
    loss.backward()
    optimizer.step()

    return train_loss


@torch.no_grad()
def evaluate_step(model, features, labels, loss_func, valid_metrics):
    model.eval()

    features = features.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        predictions = model(features)
        loss = loss_func(predictions, labels)
        valid_metrics.update(predictions, labels)
        metrics = valid_metrics.compute()
    val_metrics = {"loss": loss, **metrics}
    return val_metrics


@torch.no_grad()
def evaluate(model, dl_val, loss_func, valid_metrics):
    model.eval()
    # -----------------bar-----------------
    evaluate_iters = len(dl_val)
    evaluate_progress = ProgressBar("Evaluate", epoch=1, total_epoch=1, iters=evaluate_iters, width=10)

    for it, (features, labels) in enumerate(dl_val):
        metrics = evaluate_step(model, features, labels, loss_func, valid_metrics)
        # ----------------------------
        evaluate_progress.update((it + 1), message=metrics)
    return {name: metric.item() for name, metric in metrics.items()}


@torch.no_grad()
def predict(model, dl):
    model.eval()
    result = torch.cat([model(t[0].to(device)) for t in dl])
    target = torch.cat([t[1] for t in dl])
    return result.data, target.data


def train(model, train_loader, val_loader, start_epoch=1, end_epoch=200, loss_func=None, train_metrics=None,
          val_metrics=None, optimizer=None, scheduler=None, regularization=None):
    print("Start Training...")
    best_acc1 = 0
    train_progress = ProgressBar("Train", total_epoch=end_epoch, iters=len(train_loader), width=25)
    val_progress = ProgressBar("Val", total_epoch=end_epoch, iters=len(val_loader), width=10)
    for epoch in range(start_epoch, end_epoch + 1):
        # -----------------------------train-------------------------------
        for it, (features, labels) in enumerate(train_loader):
            train_log = train_step(model, features, labels, loss_func, train_metrics, optimizer,
                                   regularization=regularization)
            # --------------bar-------------
            train_progress.update(it + 1, epoch, train_log)
        train_metrics.reset()
        # ---------------------------------val-----------------------------------------
        for it, (features, labels) in enumerate(val_loader):
            val_log = evaluate_step(model, features, labels, loss_func, val_metrics)
            # --------------bar-------------
            val_progress.update(it + 1, epoch, val_log)
        val_metrics.reset()
        # -------update lr---------
        if scheduler:
            metric = val_log['acc'] if scheduler.mode == "max" else val_log['val_loss']
            scheduler.step(metric)
        # -------------save log -----------------
        log_dic = {"epoch": epoch, "train": train_log, "val": val_log}
        save_log_to_csv(log_dic, start_time=start_time)
        # ----------save checkpoint----------------------------
        val_acc = val_log['acc']
        if is_best := val_acc > best_acc1:
            best_acc1 = val_acc
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),

        }
        save_checkpoint(state, model_dir=train_config("save_dir"), is_best=is_best, start_time=start_time)


def main():
    global device, train_config, val_config, weight_decay, reg_loss, start_time
    train_config = Config(mode="train")
    val_config = Config(mode="val")
    device = torch.device("cuda:0") if train_config("use_gpu") and torch.cuda.is_available() else "cpu"

    train_metrics = get_metrics(num_classes=train_config("num_classes"))
    val_metrics = get_metrics(num_classes=val_config("num_classes"))
    train_metrics.to(device)
    val_metrics.to(device)

    timestamp = time.time()
    datetime_struct = datetime.datetime.fromtimestamp(timestamp)
    start_time = datetime_struct.strftime('%Y-%m-%d-%H-%M-%S-%f')
    #
    ds_train = datasets(mode="train", files=train_config("train_data_path"))

    ds_valid = datasets(mode="val", files=val_config("val_data_path"))

    trainLoader = DataLoader(ds_train, batch_size=train_config("batch_size"), shuffle=True,
                             num_workers=train_config("num_workers"), drop_last=True, prefetch_factor=4,
                             persistent_workers=True)
    validLoader = DataLoader(ds_valid, batch_size=val_config("batch_size"), shuffle=True,
                             num_workers=val_config("num_workers"), drop_last=True, prefetch_factor=4,
                             persistent_workers=True)

    # ------ load model ------------
    model = Resnet(num_classes=train_config("num_classes")).to(device)

    # ----------is  Load pretrained  --------
    if train_config("pretrain_models"):
        print("Load pretrained ...")
        state = torch.load(train_config("model_path"))
        model.load_state_dict(state['state_dict'], strict=False)

    # optimizer.load_state_dict(state['optimizer'])

    # -----------loss  and scheduler and Regularization---------------
    loss = nn.CrossEntropyLoss()

    if train_config("weight_decay") > 0:
        regularization = Regularization(model, train_config("weight_decay"), p=2).to(device)
    else:
        regularization = None
        print("No regularization")

    # ----------start training-------------
    if train_config("freeze_epoch"):
        print("Start freeze train...")
        # freeze
        for p in model.named_parameters():
            # print(p[1].requires_grad)
            if "fc" not in p[0]:
                p[1].requires_grad = False
                # print(p[0], p[1].requires_grad)
        optimizer = torch.optim.Adam(
            [{'params': [param for name, param in model.named_parameters() if 'fc' not in name]}], lr=0.001)

        # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=7)
        train(model, trainLoader, validLoader, start_epoch=train_config("start_epoch"),
              end_epoch=train_config("freeze_epoch"), loss_func=loss, train_metrics=train_metrics,
              val_metrics=val_metrics, optimizer=optimizer, scheduler=None, regularization=regularization)
        # Un freeze
        print("Start unfreeze train...")
        for p in model.named_parameters():
            # print(p[1].requires_grad)
            if "fc" not in p[0]:
                p[1].requires_grad = True

        optimizer.add_param_group({'params': [param for name, param in model.named_parameters() if 'fc' in name]})
        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)
        train(model, trainLoader, validLoader, start_epoch=train_config("freeze_epoch") + 1,
              end_epoch=train_config("end_epoch"), loss_func=loss, train_metrics=train_metrics, val_metrics=val_metrics,
              optimizer=optimizer, scheduler=scheduler, regularization=regularization)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config("learning_rate"))
        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)
        train(model, trainLoader, validLoader, start_epoch=train_config("start_epoch"),
              end_epoch=train_config("end_epoch"), loss_func=loss, train_metrics=train_metrics, val_metrics=val_metrics,
              optimizer=optimizer, scheduler=scheduler, regularization=regularization)

    # results = evaluate(model, testLoader, loss_func=Loss)


if __name__ == '__main__':
    main()
