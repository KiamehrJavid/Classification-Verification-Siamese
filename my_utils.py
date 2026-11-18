import numpy as np
import os, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_yellow = '#D0E000'
my_red = '#A02000'
my_grey = '#303730'
my_blue = '#0020A0'




def split_by_labels(dataset, label_type="make", ratios=(0.7,0.15,0.15)):
    if label_type == "make":
        labels = list(dataset.make_ids)
    else:
        labels = list(dataset.model_ids)

    random.seed(42)
    random.shuffle(labels)
    n = len(labels)
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)

    train_labels = set(labels[:n_train])
    val_labels   = set(labels[n_train:n_train+n_val])
    test_labels  = set(labels[n_train+n_val:])

    def select(indices, label_set, label_type):
        if label_type == "make":
            return [i for i in indices if dataset.items[i]["make_id"] in label_set]
        else:
            return [i for i in indices if dataset.items[i]["model_id"] in label_set]

    all_indices = list(range(len(dataset)))
    train_idx = select(all_indices, train_labels, label_type)
    val_idx   = select(all_indices, val_labels, label_type)
    test_idx  = select(all_indices, test_labels, label_type)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

class ResizePad:
    """
    Resize image keeping aspect ratio, then pad with black to target size.
    """
    def __init__(self, size=(128,128), fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image):
        target_h, target_w = self.size
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (target_w, target_h), color=self.fill)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img


def _safe_crop(img, bbox):
    if bbox is None: 
        return img
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


class SiameseDatasetBBox(Dataset):
    """
    Siamese dataset with bounding-box cropping and optional same-view pairing.

    Args:
        image_root: root folder with images (image/make/model/year/xxx.jpg)
        label_root: root folder with labels (label/make/model/year/xxx.txt)
        transform: applied AFTER bbox crop
        same_view: if True, enforce both images to have same view (when available)
        mode: 'make' or 'model' verification
    """
    def __init__(self, image_root, label_root, transform=None, same_view=False, mode="model"):
        self.image_root = image_root
        self.label_root = label_root
        self.same_view = same_view
        self.transform = transform or T.Compose([ResizePad((128,128)), T.ToTensor()])
        assert mode in ["make", "model"], "mode must be 'make' or 'model'"
        self.mode = mode

        self.items = []   # each: {'img_path','make','make_id','model','model_id','view','bbox'}

        # assign ids
        self.make_to_id = {}
        self.model_to_id = {}

        # traverse dataset
        for make in sorted(os.listdir(image_root)):
            make_dir = os.path.join(image_root, make)
            if not os.path.isdir(make_dir): continue
            if make not in self.make_to_id:
                self.make_to_id[make] = len(self.make_to_id)

            for model in sorted(os.listdir(make_dir)):
                model_dir = os.path.join(make_dir, model)
                if not os.path.isdir(model_dir): continue
                if (make, model) not in self.model_to_id:
                    self.model_to_id[(make, model)] = len(self.model_to_id)

                for year in sorted(os.listdir(model_dir)):
                    year_dir = os.path.join(model_dir, year)
                    if not os.path.isdir(year_dir): continue
                    for fn in sorted(os.listdir(year_dir)):
                        if not fn.lower().endswith((".jpg",".jpeg",".png")):
                            continue
                        img_path = os.path.join(year_dir, fn)

                        # label path (replace ext with .txt)
                        rel = os.path.relpath(img_path, image_root)
                        lbl_rel = os.path.splitext(rel)[0] + ".txt"
                        lbl_path = os.path.join(label_root, lbl_rel)

                        view, bbox = None, None
                        if os.path.isfile(lbl_path):
                            try:
                                with open(lbl_path, "r") as fh:
                                    lines = [l.strip() for l in fh if l.strip()]
                                if len(lines) >= 1:
                                    view = int(lines[0].split()[0])
                                bbox_line = None
                                if len(lines) >= 3:
                                    bbox_line = lines[2]
                                elif len(lines) >= 2 and len(lines[1].split()) >= 4:
                                    bbox_line = lines[1]
                                if bbox_line:
                                    toks = bbox_line.split()
                                    if len(toks) >= 4:
                                        bbox = list(map(int, toks[:4]))
                            except Exception:
                                pass

                        self.items.append({
                            "img_path": img_path,
                            "make": make,
                            "make_id": self.make_to_id[make],
                            "model": model,
                            "model_id": self.model_to_id[(make, model)],
                            "view": view,
                            "bbox": bbox
                        })

        self.make_to_indices = {}
        self.model_to_indices = {}
        self.make_view_to_indices = {}
        self.model_view_to_indices = {}

        for idx, it in enumerate(self.items):
            m_id = it["make_id"]
            mdl_id = it["model_id"]
            v = it["view"]

            self.make_to_indices.setdefault(m_id, []).append(idx)
            self.model_to_indices.setdefault(mdl_id, []).append(idx)

            if v is not None:
                self.make_view_to_indices.setdefault((m_id, v), []).append(idx)
                self.model_view_to_indices.setdefault((mdl_id, v), []).append(idx)

        self.make_ids = list(self.make_to_indices.keys())
        self.model_ids = list(self.model_to_indices.keys())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it1 = self.items[idx]
        img1 = Image.open(it1["img_path"]).convert("RGB")
        img1 = _safe_crop(img1, it1["bbox"])
        if self.transform: img1 = self.transform(img1)

        if self.mode == "make":
            L1 = it1["make_id"]
            view = it1["view"]
            label_pool = self.make_to_indices
            label_view_pool = self.make_view_to_indices
            labels = self.make_ids
        else:
            L1 = it1["model_id"]
            view = it1["view"]
            label_pool = self.model_to_indices
            label_view_pool = self.model_view_to_indices
            labels = self.model_ids

        if random.random() < 0.5:
            # positive
            if (self.same_view>random.random()) and view is not None:
                candidates = [i for i in label_view_pool.get((L1, view), []) if i != idx]
                if not candidates:
                    candidates = [i for i in label_pool[L1] if i != idx]
            else:
                candidates = [i for i in label_pool[L1] if i != idx]
            if not candidates:
                idx2 = idx
            else:
                idx2 = random.choice(candidates)
            label = 1
        else:
            # negative
            if (self.same_view>random.random()) and view is not None:
                candidate_labels = [lab for lab in labels if lab != L1 and (lab, view) in label_view_pool]
                if candidate_labels:
                    neg_lab = random.choice(candidate_labels)
                    idx2 = random.choice(label_view_pool[(neg_lab, view)])
                else:
                    neg_lab = random.choice([lab for lab in labels if lab != L1])
                    idx2 = random.choice(label_pool[neg_lab])
            else:
                neg_lab = random.choice([lab for lab in labels if lab != L1])
                idx2 = random.choice(label_pool[neg_lab])
            label = 0

        it2 = self.items[idx2]
        img2 = Image.open(it2["img_path"]).convert("RGB")
        img2 = _safe_crop(img2, it2["bbox"])
        if self.transform: img2 = self.transform(img2)

        return img1, img2, label


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=256, head_hidden=None, head_depth=1,
                 pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        in_dim = base.fc.in_features

        if freeze_backbone:
            for p in self.feature.parameters():
                p.requires_grad = False

        # head
        layers, dim_in = [], in_dim
        for i in range(head_depth - 1):
            dim_out = head_hidden or in_dim
            layers += [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, embedding_dim))
        self.head = nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # embeddings: z1, z2 (batch, d)
        dist = F.pairwise_distance(z1, z2)  # (batch,...)
        loss_pos = 1*label * dist.pow(2)                # same so minimize distance
        loss_neg = 1*(1 - label) * F.relu(self.margin - dist).pow(2)  # diff so push apart
        return torch.mean(loss_pos + loss_neg)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.float().to(device)

        optimizer.zero_grad()
        z1, z2 = model(img1, img2)
        loss = criterion(z1, z2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*img1.size(0)  

    return running_loss / len(dataloader.dataset)


def train(model, train_loader, criterion, optimizer, device, val_loader = None, epochs=5, verbose=True):
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        else:
            val_loss, val_acc = None, None

        if verbose: print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss} | Val Acc: {val_acc}")

    return train_losses, val_losses, val_accs        


def evaluate(model, dataloader, criterion, device, threshold=.55, pprob=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in dataloader:

            rnd = np.random.rand()
            if rnd < pprob:
                plt.imshow(img1[0].permute(1, 2, 0))
                plt.show()
                plt.imshow(img2[0].permute(1, 2, 0))
                plt.show()
                print(label[0])

            img1, img2, label = img1.to(device), img2.to(device), label.float().to(device)
            z1, z2 = model(img1, img2)
            loss = criterion(z1, z2, label)
            running_loss += loss.item() * img1.size(0)

            dist = torch.nn.functional.pairwise_distance(z1, z2)
            preds = (dist < threshold).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

            if rnd < pprob:
                print(dist[0].cpu().numpy(), label[0].numpy())
                
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def plot_distance_distribution(model, val_loader, device, save_path=None, bins=50, title=None):
    model.eval()
    dists_pos = []
    dists_neg = []
    with torch.no_grad():
        for batch in val_loader:

            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            z1, z2 = model(img1, img2)
            d = F.pairwise_distance(z1, z2)
            d = d.cpu().numpy()
            labs = label.cpu().numpy()
            dists_pos.append(d[labs == 1])
            dists_neg.append(d[labs == 0])

    if len(dists_pos) == 0: dists_pos = np.array([])
    else: dists_pos = np.concatenate(dists_pos)
    if len(dists_neg) == 0: dists_neg = np.array([])
    else: dists_neg = np.concatenate(dists_neg)

    def stats(arr):
        return float(np.mean(arr)), float(np.std(arr)), int(len(arr))
    mean_p, std_p, n_p = stats(dists_pos) if dists_pos.size else (np.nan, np.nan, 0)
    mean_n, std_n, n_n = stats(dists_neg) if dists_neg.size else (np.nan, np.nan, 0)
    print(f"Pos: n={n_p}, mean={mean_p:.4f}, std={std_p:.4f}")
    print(f"Neg: n={n_n}, mean={mean_n:.4f}, std={std_n:.4f}")


    y = np.concatenate([np.ones_like(dists_pos), np.zeros_like(dists_neg)])
    scores = np.concatenate([ -dists_pos, -dists_neg ])
    auc = roc_auc_score(y, scores)
    print(f"ROC AUC: {auc:.4f}")

    # find best threshold
    all_d = np.concatenate([dists_pos, dists_neg])
    all_y = np.concatenate([np.ones_like(dists_pos), np.zeros_like(dists_neg)])
    if all_d.size:
        thresholds = np.linspace(all_d.min(), all_d.max(), 300)
        best_acc = 0.0
        best_t = thresholds[0]
        for t in thresholds:
            preds = (all_d <= t).astype(int)
            acc = (preds == all_y).mean()
            if acc > best_acc:
                best_acc = acc
                best_t = t
        print(f"Best threshold={best_t:.4f}, Best accuracy={best_acc:.4f}")
    else:
        print("No pairs collected; cannot compute threshold/accuracy.")

    # recall for best threshold
    if all_d.size:
        preds = (all_d <= best_t).astype(int)
        tp = ((preds == 1) & (all_y == 1)).sum()
        fn = ((preds == 0) & (all_y == 1)).sum()
        recall = tp / (tp + fn)
        print(f"Best recall={recall:.4f}")



    pos_median = np.median(dists_pos) if dists_pos.size else np.nan
    neg_median = np.median(dists_neg) if dists_neg.size else np.nan


    plt.figure(figsize=(7,4))
    if dists_pos.size:
        plt.hist(dists_pos, bins=bins, alpha=0.6, label=f"same", color=my_blue)
    if dists_neg.size:
        plt.hist(dists_neg, bins=bins, alpha=0.6, label=f"different", color=my_red)

    plt.axvline(pos_median, color=my_blue, linestyle='dashed', linewidth=1, label=f"median={pos_median:.4f}")
    plt.axvline(neg_median, color=my_red, linestyle='dashed', linewidth=1, label=f"median={neg_median:.4f}")
    plt.axvline(best_t, color='green', linestyle='dashed', linewidth=1, label=f"threshold={best_t:.4f}")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Count")
    if title is None: title = "Distance distribution: same vs different"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved histogram to {save_path}")
    plt.show()
    print(f'distance between medians: {np.abs(pos_median - neg_median):.4f}', '\n')
    return (best_t)


def load_model(load_path=None, resume_training=False):
    if load_path is None:
        load_path = "/home/kia/Desktop/PoD/NNDL/models/siamese_contrastive_RN18_256_128x128.pth"
    assert os.path.exists(load_path), f"File not found: {load_path}"
    print(f"Loading model from {load_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(load_path, map_location="cpu")
    loaded_model = SiameseNet(embedding_dim=checkpoint["embedding_dim"])
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)

    if resume_training:
        optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return loaded_model, optimizer, checkpoint

    return loaded_model, checkpoint




