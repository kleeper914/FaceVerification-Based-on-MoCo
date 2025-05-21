import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc, roc_curve, f1_score
from models.inception_v1 import InceptionV1

config = {
    "valid_root_dir": "./data/data/mtcnn_lfw/",
    "pairs_file": "./data/data/lfw_test_pairs.txt",
    "batch_size": 32,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

class LFWDataset(Dataset):
    def __init__(self, img_q_path, img_k_path, labels, transform=None):
        self.img_q_path = img_q_path
        self.img_k_path = img_k_path
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.img_q_path)

    def __getitem__(self, idx):
        img_q = Image.open(self.img_q_path[idx]).convert("RGB")
        img_k = Image.open(self.img_k_path[idx]).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            img_q = self.transform(img_q)
            img_k = self.transform(img_k)
        return img_q, img_k, label
    
def read_pairs(pairs_file):
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

def load_valid_data(valid_root_dir, pairs_file):
    pairs = read_pairs(pairs_file)
    image_q_path = []
    image_k_path = []
    for i in range(len(pairs)):
        if len(pairs[i]) == 3:
            image_q_path.append(valid_root_dir + pairs[i][0] + "/" + pairs[i][0] + "_" + "%04d" % int(pairs[i][1]) + ".jpg")
            image_k_path.append(valid_root_dir + pairs[i][0] + "/" + pairs[i][0] + "_" + "%04d" % int(pairs[i][2]) + ".jpg")
        elif len(pairs[i]) == 4:
            image_q_path.append(valid_root_dir + pairs[i][0] + "/" + pairs[i][0] + "_" + "%04d" % int(pairs[i][1]) + ".jpg")
            image_k_path.append(valid_root_dir + pairs[i][2] + "/" + pairs[i][2] + "_" + "%04d" % int(pairs[i][3]) + ".jpg")

    labels = []
    for i in range(len(pairs)):
        if len(pairs[i]) == 3:
            labels.append(1)
        elif len(pairs[i]) == 4:
            labels.append(0)
    
    valid_dataset = LFWDataset(image_q_path, image_k_path, labels, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    return valid_loader

def evaluate(model, valid_loader, device, log_path):
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for i, (img_q, img_k, label) in tqdm(enumerate(valid_loader)):
            img_q, img_k = img_q.to(device), img_k.to(device)
            out_q = model(img_q)
            out_k = model(img_k)
            out = F.cosine_similarity(out_q, out_k)
            labels += label.cpu().numpy().tolist()
            scores += out.cpu().numpy().tolist()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_index]
        accu = accuracy_score(labels, [1 if score >= optimal_threshold else 0 for score in scores])

        print(f"AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accu:.4f}")
        #print(f"Optimal Threshold: {optimal_threshold:.4f}")
        with open(log_path, "a") as f:
            f.write(
                f"AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accu:.4f}\n"
            )
    
    return auc_score, optimal_threshold, accu

def main():
    # Load the model
    model = InceptionV1(classify=False)
    # Load the pre-trained weights
    model_path = "./2025-04-23/model_best.pth.tar"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(config["device"])

    # Load the validation data
    valid_loader = load_valid_data(config["valid_root_dir"], config["pairs_file"])

    # Evaluate the model
    # 创建txt文件存储训练过程
    current_time = time.strftime("%Y-%m-%d", time.localtime())
    log_name = f"{current_time}_log.txt"
    if not os.path.exists(current_time):
        os.makedirs(current_time)
    log_path = os.path.join(current_time, log_name)
    auc_score, optimal_threshold, accuracy = evaluate(model, valid_loader, config["device"], log_path)
    print(f"AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Evaluation completed. Results saved to {log_path}")

if __name__ == "__main__":
    main()