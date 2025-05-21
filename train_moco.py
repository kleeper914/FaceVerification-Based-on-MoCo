import argparse
import numpy as np
import random
import shutil
import time
import math
import torch
from torch import nn
from torch.nn import functional as F
#from torchvision import models
from models import ResNet34, ResNet50, InceptionV1
from sklearn.metrics import auc, accuracy_score, roc_curve
import os
from moco.model import MoCo
from manager import AverageMeter, ProgressMeter
from dataset import get_pretrain_dataloader, get_lfw_dataloader

# 备选模型
model_names = ["resnet34", "resnet50", "inceptionv1"]

parser = argparse.ArgumentParser(description="PyTorch MoCo Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--model",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--seed",
    default=2025,
    type=int,
    help="seed for initializing training. (default: 2025)",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="number of total epochs to run (default: 200)",
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="mini-batch size (default: 32)",
)
parser.add_argument(
    "--lr",
    default=0.03,
    type=float,
    help="initial learning rate (default: 0.03)",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    help="momentum of SGD solver (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    default=1e-4,
    type=float,
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--dim",
    default=128,
    type=int,
    help="feature dimension (default: 128)",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed():
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.cuda.manual_seed_all(params["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch):
    # 采用cosine annealing
    new_lr = params["lr"]
    new_lr *= 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def save_checkpoint(state, is_best):
    epoch = state["epoch"]
    date = time.strftime("%Y-%m-%d", time.localtime())
    file_path = "MoCo-{}-{}".format(params["model"], date)
    filename = f"checkpoint_{epoch}.pth.tar"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save(state, os.path.join(file_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(file_path, filename),
            os.path.join(file_path, "{}_best.pth.tar".format(params["model"]))
        )

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0   

def base_encoder(num_classes=128):
    if params["model"] == "resnet34":
        model = ResNet34(num_classes=num_classes)
    elif params["model"] == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif params["model"] == "inceptionv1":
        model = InceptionV1(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")
    return model

def train_one_epoch(model, train_loader, optimizer, criterion, epoch, log_path):
    model.train()
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )
    end = time.time()
    for i, (images) in enumerate(train_loader):
        # 数据加载时间
        data_time.update(time.time() - end)
        images = [im.to(device) for im in images]
        
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学习率
        learning_rates.update(optimizer.param_groups[0]["lr"])
        
        # 计算一个batch的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % params['print_freq'] == 0:
            progress.display(i)
            #print(f"loss: {losses.avg:.4f}, acc1: {top1.avg:.2f}, acc5: {top5.avg:.2f}")
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"LR: {learning_rates.val:.4e} ({learning_rates.avg:.4e})\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                    f"Acc@5 {top5.val:.2f} ({top5.avg:.2f})\n"
                )

def valid(model, valid_loader, log_path):
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for i, (img_q, img_k, label) in enumerate(valid_loader):
            img_q, img_k = img_q.to(device), img_k.to(device)
            out_q = model.encoder_q(img_q)
            out_k = model.encoder_q(img_k)
            out_q = F.normalize(out_q, dim=1)
            out_k = F.normalize(out_k, dim=1)
            out = F.cosine_similarity(out_q, out_k)
            labels += label.cpu().numpy().tolist()
            scores += out.cpu().numpy().tolist()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_index]
        accuracy = accuracy_score(labels, [1 if score >= optimal_threshold else 0 for score in scores])

        print(f"AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accuracy:.4f}")
        #print(f"Optimal Threshold: {optimal_threshold:.4f}")
        with open(log_path, "a") as f:
            f.write(
                f"AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accuracy:.4f}\n"
            )
    
    return auc_score, optimal_threshold, accuracy

def train(model, train_loader, valid_loader, optimizer, criterion):
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(params["epochs"]):
        adjust_learning_rate(optimizer, epoch)

        # 创建txt文件存储训练过程
        current_time = time.strftime("MoCo-%Y-%m-%d", time.localtime())
        log_name = f"{current_time}_log.txt"
        if not os.path.exists(current_time):
            os.makedirs(current_time)
        log_path = os.path.join(current_time, log_name)

        train_one_epoch(model, train_loader, optimizer, criterion, epoch, log_path)
        auc_score, optimal_threshold, accu = valid(model, valid_loader, log_path)
        early_stopping(accu)
        # 保存模型
        is_best = accu > best_acc
        best_acc = max(accu, best_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "base_encoder": params["model"],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            #filename=os.path.join(current_time, "checkpoint_{:04d}.pth.tar".format(epoch))
        )
        if early_stopping.early_stop:
            print("Early stopping !!!")
            break

def main() -> None:
    args = parser.parse_args()
    global params
    params = {
        "data_path": args.data,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "print_freq": 20,
    }
    casia_root_dir = os.path.join(args.data, "casia-webface")
    lfw_root_dir = os.path.join(args.data, "mtcnn_lfw")
    pairs_file = os.path.join(args.data, "lfw_test_pairs.txt")
    set_seed()
    print("Using device: ", device)
    print("=> creating model '{}'".format(params["model"]))
    model = MoCo(
        base_encoder=base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=True,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["lr"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss().to(device)
    print("=> creating data loaders")
    train_loader = get_pretrain_dataloader(casia_root_dir, params["batch_size"])
    valid_loader = get_lfw_dataloader(lfw_root_dir, pairs_file, params["batch_size"])
    print("=> start training")
    train(model, train_loader, valid_loader, optimizer, criterion)
    print("=> training finished")

if __name__ == "__main__":
    main()