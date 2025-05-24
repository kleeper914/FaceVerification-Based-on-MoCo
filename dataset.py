import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from pathlib import Path
import os
import random
import moco.loader

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomApply([moco.loader.Solarization()], p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

class PretrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(Path(root_dir).glob("**/*.jpeg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        view1, view2 = self.transform(img), self.transform(img)
        return view1, view2

def read_pairs(pairs_file):
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

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
    
def get_pretrain_dataloader(root_dir, batch_size):
    train_dataset = PretrainDataset(root_dir, transform=augmentation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return train_loader

def get_lfw_dataloader(valid_root_dir, pairs_file, batch_size):
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

    valid_dataset = LFWDataset(
        image_q_path=image_q_path,
        image_k_path=image_k_path,
        labels=labels,
        transform=valid_transform
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return valid_loader

class FacePairDataset(Dataset):
    def __init__(self, image_folder, num_pairs=10000, transform=None):
        self.image_folder = image_folder
        self.num_pairs = num_pairs
        self.transform = transform
        self.person_to_images = self._build_person_dict()
        self.pairs = self._generate_pairs(num_pairs)
        
    def _build_person_dict(self):
        person_dict = {}
        image_folds = os.listdir(self.image_folder)[1:] # for windows
        for person in image_folds:
            person_path = os.path.join(self.image_folder, person)
            if os.path.isdir(person_path):
                imgs = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.endswith(".jpeg") or img.endswith(".jpg")]
                if len(imgs) > 0:   # 至少两张图片才能生成anchor-positive
                    person_dict[person] = imgs
        return person_dict
    
    def _generate_pairs(self, num_pairs=10000):
        pairs = []
        person = list(self.person_to_images.keys())
        for _ in range(num_pairs):
            anchor_p = random.choice(person)
            positive = random.sample(self.person_to_images[anchor_p], 2)
            anchor_img = positive[0]
            positive_img = positive[1]
            # 负样本来自不同的人
            negative_p = random.choice(person)
            while negative_p == anchor_p:
                negative_p = random.choice(person)
            negative_img = random.choice(self.person_to_images[negative_p])

            pairs.append((anchor_img, positive_img, negative_img))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.pairs[idx]
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative

augmentation_triplet = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

def get_triplet_dataloader(image_folder, num_pairs=10000, batch_size=32):
    dataset = FacePairDataset(
        image_folder=image_folder,
        num_pairs=num_pairs,
        transform=augmentation_triplet
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader

# 配合semi-hard triplet selection策略使用
class FacePairSemiHardDataset(Dataset):
    def __init__(self, image_folder, num_pairs=10000, transform=None):
        self.image_folder = image_folder
        self.num_pairs = num_pairs
        self.transform = transform
        self.image_paths = []
        self.labels = []

        #person_folders = os.listdir(image_folder)[1:] 
        person_folders = os.listdir(image_folder)
        for idx, person in enumerate(person_folders):
            person_path = os.path.join(image_folder, person)
            if os.path.isdir(person_path):
                imgs = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.endswith(".jpeg") or img.endswith(".jpg")]
                self.image_paths += imgs
                self.labels += [idx] * len(imgs)    # use integer labels for each person

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_semi_hard_triplet_dataloader(image_folder, num_pairs=10000, batch_size=32):
    dataset = FacePairSemiHardDataset(
        image_folder=image_folder,
        num_pairs=num_pairs,
        transform=augmentation_triplet
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader
    
def pairwise_distance(embeddings):
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0.0)
    return distances

def get_semi_hard_triplets(embeddings, labels, margin=0.2):
    distance_matrix = pairwise_distance(embeddings)
    labels = labels.unsqueeze(1)
    mask_anchor_positive = (labels == labels.t()) & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    mask_anchor_negative = (labels != labels.t())

    triplets = []

    for i in range(len(embeddings)):
        anchor = embeddings[i]
        pos_indices = torch.where(mask_anchor_positive[i])[0]
        neg_indices = torch.where(mask_anchor_negative[i])[0]

        for pos_idx in pos_indices:
            ap_dist = distance_matrix[i][pos_idx]
            semi_hard_negatives = neg_indices[
                (distance_matrix[i][neg_indices] < ap_dist + margin) &
                (distance_matrix[i][neg_indices] > ap_dist)
            ]
            if len(semi_hard_negatives) > 0:
                neg_idx = semi_hard_negatives[
                    torch.randint(len(semi_hard_negatives), (1,)).item()
                ]
                triplets.append((i, pos_idx.item(), neg_idx.item()))

    return triplets

if __name__ == "__main__":
    # 测试pairwise_distance函数
    embeddings = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    distances = pairwise_distance(embeddings)
    print("Pairwise distances:")
    print(distances)