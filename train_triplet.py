import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import auc, accuracy_score, roc_curve
from tqdm import tqdm
import os
import time
import math
import random
import shutil
from dataset import get_semi_hard_triplets, get_semi_hard_triplet_dataloader, get_lfw_dataloader
from losses import TripletLoss
from manager import AverageMeter, ProgressMeter

model_names = ["resnet34", "resnet50", "inceptionv1"]

parser = argparse.ArgumentParser(description="PyTorch Triplet Loss Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--model",
    default="inceptionv1",
    choices=model_names,
    help="model architecture: | ".join(model_names) + " (default: inceptionv1)",
)
parser.add_argument(
    "--model-path",
    default="weights/inceptionv1.pth",
    type=str,
    help="path to MoCo pre-trained model (default: weights/inceptionv1.pth)",
)
parser.add_argument(
    "--dim",
    default=128,
    type=int,
    help="feature dimension (default: 128)",
)
parser.add_argument(
    "--seed",
    default=2025,
    type=int,
    help="seed for initializing training. (default: 2025)",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="number of total epochs to run (default: 100)",
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="mini-batch size (default: 32)",
)
parser.add_argument(
    "--lr",
    default=0.01,
    type=float,
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--weight-decay",
    default=1e-4,
    type=float,
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--num-pairs",
    default=10000,
    type=int,
    help="number of triplet pairs to sample from the dataset (default: 10000)",
)
parser.add_argument(
    "--margin",
    default=0.2,
    type=float,
    help="margin for triplet loss (default: 0.2)",
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

def save_checkpoint(state, is_best):
    epoch = state["epoch"]
    date = time.strftime("%Y-%m-%d", time.localtime())
    file_path = "Triplet-{}-{}".format(params["model"], date)
    filename = f"checkpoint_{epoch}.pth.tar"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save(state, os.path.join(file_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(file_path, filename),
            os.path.join(file_path, "model_best.pth.tar")
        )

def adjust_learning_rate(optimizer, epoch):
    # 采用cosine annealing
    new_lr = params["lr"]
    new_lr *= 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

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

def train_one_epoch(model, train_loader, optimizer, criterion, epoch, log_path):
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    print(f"Epoch: {epoch}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)

        triplets = get_semi_hard_triplets(embeddings, labels)
        if len(triplets) == 0:
            continue
        anchor = torch.stack([embeddings[a] for a, _, _ in triplets])
        positive = torch.stack([embeddings[p] for _, p, _ in triplets])
        negative = torch.stack([embeddings[n] for _, _, n in triplets])

        optimizer.zero_grad()
        loss = criterion(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        if i % params["print_freq"] == 0:
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"LR {optimizer.param_groups[0]['lr']:.4e}\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\n"
                )
    progress.display(i)
    print(f"loss: {losses.avg:.4f}")

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

        current_time = time.strftime("Triplet-%Y-%m-%d", time.localtime())
        log_name = f"{current_time}_log.txt"
        if not os.path.exists(current_time):
            os.makedirs(current_time)
        log_path = os.path.join(current_time, log_name)
    
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, log_path)
        auc_score, optimal_threshold, accu = valid(model, valid_loader, log_path)
        early_stopping(accu)

        is_best = accu > best_acc
        best_acc = max(accu, best_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model": params["model"],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )
        if early_stopping.early_stop:
            print("Early stopping")
            break

def create_model(model_name):
    if model_name == "resnet34":
        from models import ResNet34
        model = ResNet34(num_classes=params["dim"])
    elif model_name == "resnet50":
        from models import ResNet50
        model = ResNet50(num_classes=params["dim"])
    elif model_name == "inceptionv1":
        from models import InceptionV1
        model = InceptionV1(num_classes=params["dim"])
    checkpoint = torch.load(params["model_path"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    return model
                
def main() -> None:
    args = parser.parse_args()
    global params
    params = {
        "data_path": args.data,
        "model": args.model,
        "model_path": args.model_path,
        "dim": args.dim,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_pairs": args.num_pairs,
        "margin": args.margin,
        "print_freq": 20,
    }
    casia_root_dir = os.path.join(args.data, "casia-webface")
    lfw_root_dir = os.path.join(args.data, "mtcnn_lfw")
    pairs_file = os.path.join(args.data, "lfw_test_pairs.txt")
    set_seed()
    print("Using device:", device)
    print("=> creating model '{}'".format(params["model"]))
    model = create_model(params["model"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    criterion = TripletLoss(margin=params["margin"]).to(device)
    print("=> creating data loader")
    train_loader = get_semi_hard_triplet_dataloader(
        casia_root_dir,
        batch_size=params["batch_size"],
        num_pairs=params["num_pairs"],
    )
    valid_loader = get_lfw_dataloader(
        lfw_root_dir,
        pairs_file,
        batch_size=params["batch_size"],
    )
    print("=> Start training")
    train(model, train_loader, valid_loader, optimizer, criterion)
    print("=> Training finished")

if __name__ == "__main__":
    main()