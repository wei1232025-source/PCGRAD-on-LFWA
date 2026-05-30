# train_fairness.py
import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import datasets

SMILE_NAMES = {"smiling", "smile"}
MALE_NAMES = {"male", "gender"}


@dataclass
class Sample:
    image_idx: int  # 本地 Arrow 数据集中的行索引
    smile: int
    male: int


class LFWAAttributeDataset(Dataset):
    """LFWA attribute dataset loaded from local disk and processed on-the-fly."""

    def __init__(self, hf_dataset, annotation_file, transform=None):
        self.hf_dataset = hf_dataset
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError("No valid LFWA samples were found.")

    def _load_samples(self):
        lines = [
            line.strip()
            for line in self.annotation_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if len(lines) < 3:
            raise ValueError("Attribute txt file is too short.")

        # 动态定位属性定义行
        header_idx = None
        for i, line in enumerate(lines):
            tokens = [t.lower().strip() for t in line.split()]
            if any("person" in t for t in tokens) and any("imagenum" in t for t in tokens):
                header_idx = i
                break

        if header_idx is None:
            header_idx = 1 if lines[0].split()[0].isdigit() else 0

        header_line = lines[header_idx]
        attr_names = [name.strip().lower() for name in header_line.lstrip("#").strip().split("\t")]
        attr_map = {name: idx for idx, name in enumerate(attr_names)}

        smile_idx = self._find_attr(attr_map, SMILE_NAMES)
        male_idx = self._find_attr(attr_map, MALE_NAMES)

        # 构建本地数据集文件名到索引的哈希表，实现 O(1) 查询
        filename_to_idx = {row["filename"]: idx for idx, row in enumerate(self.hf_dataset)}

        samples = []
        for line in lines[header_idx + 1 :]:
            parts = line.split("\t")
            if len(parts) < len(attr_names):
                continue
            person = parts[0].strip()
            imagenum = parts[1].strip()

            # 格式化拼装 LFW 标准文件名
            person_formatted = person.replace(" ", "_")
            imagenum_formatted = f"{int(imagenum):04d}"
            filename = f"{person_formatted}_{imagenum_formatted}.jpg"

            if filename not in filename_to_idx:
                continue

            try:
                smile_val = float(parts[smile_idx])
                male_val = float(parts[male_idx])
            except ValueError:
                continue

            samples.append(
                Sample(
                    image_idx=filename_to_idx[filename],
                    smile=1 if smile_val > 0 else 0,
                    male=1 if male_val > 0 else 0,
                )
            )
        return samples

    @staticmethod
    def _find_attr(attr_map, candidates):
        for candidate in candidates:
            if candidate in attr_map:
                return attr_map[candidate]
        raise ValueError(f"Missing required attribute. Expected one of: {sorted(candidates)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 从本地加载的 datasets 数据中获取 PIL Image 对象
        image = self.hf_dataset[sample.image_idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            torch.tensor(sample.smile, dtype=torch.float32),
            torch.tensor(sample.male, dtype=torch.float32),
        )


class FaceFairnessNet(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.smile_head = nn.Linear(feature_dim, 1)
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        smile_logits = self.smile_head(features).squeeze(1)
        gender_logits = self.gender_head(features).squeeze(1)
        return features, smile_logits, gender_logits


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_accuracy(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


def balanced_accuracy_from_logits(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).long()
    targets = targets.long()
    pos = targets == 1
    neg = targets == 0
    tpr = (preds[pos] == 1).float().mean().item() if pos.any() else 0.0
    tnr = (preds[neg] == 0).float().mean().item() if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def flatten_grads(grads, params):
    flat = []
    for grad, param in zip(grads, params):
        flat.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(flat)


def assign_flat_grad(params, flat_grad):
    offset = 0
    for param in params:
        size = param.numel()
        param.grad = flat_grad[offset : offset + size].view_as(param).clone()
        offset += size


def pcgrad_two_task(g_smile, g_degender):
    dot = torch.dot(g_smile, g_degender)
    smile_norm = torch.norm(g_smile) + 1e-12
    degender_norm = torch.norm(g_degender) + 1e-12
    cosine = (dot / (smile_norm * degender_norm)).item()
    conflict = dot.item() < 0
    if conflict:
        original_smile = g_smile
        original_degender = g_degender
        g_smile = original_smile - dot / (degender_norm**2) * original_degender
        g_degender = original_degender - dot / (smile_norm**2) * original_smile
    return g_smile + g_degender, cosine, conflict


def train_epoch(
    model,
    loader,
    main_optimizer,
    gender_optimizer,
    smile_criterion,
    gender_criterion,
    device,
    mode,
    adv_weight,
):
    model.train()
    totals = {
        "loss_smile": 0.0,
        "loss_gender": 0.0,
        "smile_acc": 0.0,
        "gender_acc": 0.0,
        "gender_bal_acc": 0.0,
        "cosine": 0.0,
        "conflict_rate": 0.0,
    }
    backbone_params = list(model.backbone.parameters())

    for images, smile_targets, gender_targets in loader:
        images = images.to(device)
        smile_targets = smile_targets.to(device)
        gender_targets = gender_targets.to(device)

        gender_optimizer.zero_grad()
        with torch.no_grad():
            features = model.backbone(images)
        gender_logits = model.gender_head(features.detach()).squeeze(1)
        gender_loss = gender_criterion(gender_logits, gender_targets)
        gender_loss.backward()
        gender_optimizer.step()

        main_optimizer.zero_grad()
        _, smile_logits, gender_logits_for_backbone = model(images)
        smile_loss = smile_criterion(smile_logits, smile_targets)
        degender_loss = -adv_weight * gender_criterion(gender_logits_for_backbone, gender_targets)

        if mode == "smile_only":
            smile_loss.backward()
            cosine = 0.0
            conflict = False
        elif mode == "naive_adv":
            (smile_loss + degender_loss).backward()
            cosine = 0.0
            conflict = False
        elif mode == "pcgrad":
            smile_grads = torch.autograd.grad(
                smile_loss, backbone_params, retain_graph=True, allow_unused=True
            )
            degender_grads = torch.autograd.grad(
                degender_loss, backbone_params, retain_graph=True, allow_unused=True
            )
            merged_grad, cosine, conflict = pcgrad_two_task(
                flatten_grads(smile_grads, backbone_params),
                flatten_grads(degender_grads, backbone_params),
            )
            smile_head_params = list(model.smile_head.parameters())
            smile_head_grads = torch.autograd.grad(
                smile_loss, smile_head_params, retain_graph=False, allow_unused=True
            )
            assign_flat_grad(backbone_params, merged_grad)
            for param, grad in zip(smile_head_params, smile_head_grads):
                param.grad = None if grad is None else grad.clone()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        main_optimizer.step()

        batch_size = images.size(0)
        totals["loss_smile"] += smile_loss.item()
        totals["loss_gender"] += gender_loss.item()
        totals["smile_acc"] += binary_accuracy(smile_logits.detach(), smile_targets) * batch_size
        totals["gender_acc"] += binary_accuracy(gender_logits.detach(), gender_targets) * batch_size
        totals["gender_bal_acc"] += (
            balanced_accuracy_from_logits(gender_logits.detach(), gender_targets) * batch_size
        )
        totals["cosine"] += cosine
        totals["conflict_rate"] += float(conflict)

    n_batches = len(loader)
    n_samples = len(loader.dataset)
    return {
        "loss_smile": totals["loss_smile"] / n_batches,
        "loss_gender": totals["loss_gender"] / n_batches,
        "smile_acc": totals["smile_acc"] / n_samples,
        "gender_acc": totals["gender_acc"] / n_samples,
        "gender_bal_acc": totals["gender_bal_acc"] / n_samples,
        "cosine": totals["cosine"] / n_batches,
        "conflict_rate": totals["conflict_rate"] / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, smile_criterion, gender_criterion, device):
    model.eval()
    totals = {"loss_smile": 0.0, "loss_gender": 0.0, "smile_acc": 0.0, "gender_acc": 0.0, "gender_bal_acc": 0.0}
    for images, smile_targets, gender_targets in loader:
        images = images.to(device)
        smile_targets = smile_targets.to(device)
        gender_targets = gender_targets.to(device)
        _, smile_logits, gender_logits = model(images)
        batch_size = images.size(0)
        totals["loss_smile"] += smile_criterion(smile_logits, smile_targets).item()
        totals["loss_gender"] += gender_criterion(gender_logits, gender_targets).item()
        totals["smile_acc"] += binary_accuracy(smile_logits, smile_targets) * batch_size
        totals["gender_acc"] += binary_accuracy(gender_logits, gender_targets) * batch_size
        totals["gender_bal_acc"] += balanced_accuracy_from_logits(gender_logits, gender_targets) * batch_size

    n_batches = len(loader)
    n_samples = len(loader.dataset)
    return {
        "loss_smile": totals["loss_smile"] / n_batches,
        "loss_gender": totals["loss_gender"] / n_batches,
        "smile_acc": totals["smile_acc"] / n_samples,
        "gender_acc": totals["gender_acc"] / n_samples,
        "gender_bal_acc": totals["gender_bal_acc"] / n_samples,
    }


def estimate_pos_weight(subset, label_name):
    positives = 0
    for index in subset.indices:
        positives += getattr(subset.dataset.samples[index], label_name)
    negatives = len(subset) - positives
    return torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])


def run_experiment(mode, train_loader, test_loader, args, device):
    print(f"\n>>> Running experiment: {mode.upper()} <<<")
    model = FaceFairnessNet().to(device)
    smile_criterion = nn.BCEWithLogitsLoss()
    gender_criterion = nn.BCEWithLogitsLoss(
        pos_weight=estimate_pos_weight(train_loader.dataset, "male").to(device)
    )
    main_optimizer = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.smile_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    gender_optimizer = torch.optim.Adam(
        model.gender_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {
        "epoch": [],
        "test_smile_acc": [],
        "test_gender_acc": [],
        "test_gender_bal_acc": [],
        "cosine": [],
        "conflict_rate": [],
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            main_optimizer,
            gender_optimizer,
            smile_criterion,
            gender_criterion,
            device,
            mode,
            args.adv_weight,
        )
        test_metrics = evaluate(model, test_loader, smile_criterion, gender_criterion, device)
        history["epoch"].append(epoch)
        history["test_smile_acc"].append(test_metrics["smile_acc"])
        history["test_gender_acc"].append(test_metrics["gender_acc"])
        history["test_gender_bal_acc"].append(test_metrics["gender_bal_acc"])
        history["cosine"].append(train_metrics["cosine"])
        history["conflict_rate"].append(train_metrics["conflict_rate"])
        print(
            f"Epoch [{epoch:02d}/{args.epochs}] | "
            f"L-Smiling: {train_metrics['loss_smile']:.3f} "
            f"L-Gender: {train_metrics['loss_gender']:.3f} | "
            f"Test Smile-Acc: {test_metrics['smile_acc'] * 100:5.1f}% "
            f"Gender-Acc: {test_metrics['gender_acc'] * 100:5.1f}% "
            f"Gender-Bal-Acc: {test_metrics['gender_bal_acc'] * 100:5.1f}% | "
            f"Cos-Sim: {train_metrics['cosine']:+.3f} "
            f"Conflict: {train_metrics['conflict_rate'] * 100:4.1f}%"
        )
    return history


def split_dataset(dataset, train_ratio, seed):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, test_size], generator=generator)


def make_loaders(args):
    # 验证本地文件是否存在
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"本地数据集未找到：'{args.data_root}'。请先运行步骤一代码。")
    if not os.path.exists(args.annotation_file):
        raise FileNotFoundError(f"标注文件未找到：'{args.annotation_file}'。请先运行步骤一代码。")

    print(f"Loading local dataset from: {args.data_root}")
    hf_dataset = datasets.load_from_disk(args.data_root)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = LFWAAttributeDataset(hf_dataset, args.annotation_file, train_transform)
    eval_dataset = LFWAAttributeDataset(hf_dataset, args.annotation_file, test_transform)

    train_subset, _ = split_dataset(train_dataset, args.train_ratio, args.seed)
    _, test_subset = split_dataset(eval_dataset, args.train_ratio, args.seed)
    return (
        DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        train_dataset,
    )


def describe_dataset(dataset):
    male = sum(sample.male for sample in dataset.samples)
    smile = sum(sample.smile for sample in dataset.samples)
    both = sum(sample.male and sample.smile for sample in dataset.samples)
    total = len(dataset)
    print("\nDataset statistics")
    print(f"Total samples: {total}")
    print(f"Male ratio: {male / total * 100:.2f}%")
    print(f"Gender majority-class baseline: {max(male, total - male) / total * 100:.2f}%")
    print(f"Smiling ratio: {smile / total * 100:.2f}%")
    print(f"Male & Smiling: {both}")
    print(f"Male & Not Smiling: {male - both}")
    print(f"Female & Smiling: {smile - both}")
    print(f"Female & Not Smiling: {total - male - smile + both}")


def plot_results(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"smile_only": "tab:blue", "naive_adv": "tab:orange", "pcgrad": "tab:red"}
    for mode, history in results.items():
        label = mode.replace("_", " ").upper()
        color = colors.get(mode)
        axes[0].plot(history["epoch"], history["test_smile_acc"], marker="o", label=label, color=color)
        axes[1].plot(history["epoch"], history["test_gender_bal_acc"], marker="o", label=label, color=color)
        axes[2].plot(history["epoch"], history["cosine"], marker="o", label=label, color=color)

    axes[0].set_title("Utility: Smiling Prediction")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Smile Accuracy")
    axes[1].set_title("Leakage: Gender Attack")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Gender Balanced Accuracy")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="Random Guess")
    axes[2].set_title("Backbone Gradient Conflict")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].axhline(0.0, color="gray", linestyle=":")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved figure: {output_path}")
    plt.show() # 在 Colab 中直接画出分析图表


def parse_args():
    parser = argparse.ArgumentParser(description="LFWA smiling utility vs gender-removal with PCGrad.")
    parser.add_argument("--data-root", default="lfw_dataset_local", help="本地化的 Hugging Face dataset 目录")
    parser.add_argument("--annotation-file", default="lfw_attributes.txt", help="下载得到的属性标注文件")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["smile_only", "naive_adv", "pcgrad"],
        choices=["smile_only", "naive_adv", "pcgrad"],
    )
    parser.add_argument("--output", default="lfwa_pcgrad_comparison.png")

    # 核心修复：使用 parse_known_args 替代 parse_args，从而忽略 Jupyter/Colab 的内部临时参数，避免报错
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()} | Active Device: {device}")

    train_loader, test_loader, full_dataset = make_loaders(args)
    describe_dataset(full_dataset)

    results = {}
    for mode in args.modes:
        results[mode] = run_experiment(mode, train_loader, test_loader, args, device)

    print("\n" + "=" * 92)
    print("FINAL LFWA MULTI-OBJECTIVE COMPARISON")
    print("=" * 92)
    print(f"{'Method':<16} | {'Smile Acc':>10} | {'Gender Acc':>10} | {'Gender Bal-Acc':>15} | {'Conflict':>9}")
    print("-" * 92)
    for mode, history in results.items():
        print(
            f"{mode.upper():<16} | "
            f"{history['test_smile_acc'][-1] * 100:9.2f}% | "
            f"{history['test_gender_acc'][-1] * 100:9.2f}% | "
            f"{history['test_gender_bal_acc'][-1] * 100:14.2f}% | "
            f"{history['conflict_rate'][-1] * 100:8.2f}%"
        )
    print("=" * 92)
    plot_results(results, args.output)