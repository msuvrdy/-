"""Experiment E: LightCNN + CBAM + SupCon + MixUp, 4-fold LOSO (unloaded, constant speed)."""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

FAULT2ID = {"H_H": 0, "R_U": 1, "R_M": 2, "S_W": 3, "V_U": 4, "B_R": 5, "K_A": 6, "F_B": 7}
ID2FAULT = {v: k for k, v in FAULT2ID.items()}
FAULT_ORDER = ["B_R", "F_B", "H_H", "K_A", "R_M", "R_U", "S_W", "V_U"]
WINDOWS_PER_FILE = {"train": 132, "val": 29, "test": 40}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "D_processed_data", "D_experiment_mel_32", "batches")


class AllWindowDataset(Dataset):
    """Load batch npz files, filter by speed/load, merge train+val+test windows."""

    def __init__(
        self,
        batch_dir,
        speed_ids,
        load_id,
        split="full",
        indices=None,
        num_batches=8,
        temp_norm="global",
    ):
        file_info = []
        for bi in range(num_batches):
            load_batch = 0 if bi < num_batches // 2 else 1
            fault_start = (bi % (num_batches // 2)) * 2
            for fj in range(2):
                fault = FAULT_ORDER[fault_start + fj]
                for sp in [1, 2, 3, 4]:
                    file_info.append({"speed": sp, "load": load_batch, "fault": fault})

        mel_list, temp_list, label_list, speed_list = [], [], [], []
        for i, meta in enumerate(file_info):
            if meta["speed"] not in speed_ids or meta["load"] != load_id:
                continue
            batch_i = i // 8
            file_i = i % 8
            data = np.load(os.path.join(batch_dir, f"batch_{batch_i:02d}.npz"))

            for seg, mel_key, label_key, temp_key in [
                ("train", "tst", "tlt", "tvt"),
                ("val", "vst", "vlt", "vvt"),
                ("test", "est", "elt", "evt"),
            ]:
                wp = WINDOWS_PER_FILE[seg]
                s, e = file_i * wp, (file_i + 1) * wp
                mel_list.append(data[mel_key][s:e])
                temp_list.append(data[temp_key][s:e])
                label_list.append(data[label_key][s:e])
                speed_list.append(np.full(e - s, meta["speed"], dtype=np.int32))

        mel_all = np.concatenate(mel_list, axis=0)
        temp_all = np.concatenate(temp_list, axis=0)
        label_all = np.concatenate(label_list, axis=0)
        speed_all = np.concatenate(speed_list, axis=0)

        if indices is not None and split != "full":
            mel_all = mel_all[indices]
            temp_all = temp_all[indices]
            label_all = label_all[indices]
            speed_all = speed_all[indices]

        self.mel_raw = mel_all
        self.temp_raw = temp_all
        self.labels = label_all
        self.speeds = speed_all

        stats = np.load(os.path.join(batch_dir, "..", "norm_stats.npz"))
        n_freq = stats["stft_mean"].size // 4
        self.mel_mean = torch.from_numpy(stats["stft_mean"]).float().view(4, n_freq, 1)
        self.mel_std = torch.from_numpy(stats["stft_std"]).float().view(4, n_freq, 1) + 1e-8
        if temp_norm == "global":
            self.temp_min = float(stats["temp_min"])
            self.temp_max = float(stats["temp_max"])
        else:
            self.temp_min = float(temp_all.min())
            self.temp_max = float(temp_all.max())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = torch.from_numpy(self.mel_raw[idx]).float()
        mel = (mel - self.mel_mean) / self.mel_std
        temp_val = (self.temp_raw[idx] - self.temp_min) / (self.temp_max - self.temp_min + 1e-8)
        temp = torch.tensor([temp_val], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        speed = torch.tensor(self.speeds[idx], dtype=torch.long)
        return mel, temp, label, speed


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.view(b, c, -1).mean(-1)
        return x * self.fc(y).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class LightCNN_CBAM_SupCon(nn.Module):
    """~552,558 parameters; cbam2/cbam3 registered but not used in forward."""

    def __init__(self, num_classes=8, in_channels=4, proj_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.cbam = CBAM(256)
        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temp_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        feat_dim = 256 + 16
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.projector = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))

    def extract_features(self, mel, temp):
        x = self.conv1(mel)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return torch.cat([x, self.temp_mlp(temp)], dim=1)

    def forward(self, mel, temp, return_embedding=False):
        feat = self.extract_features(mel, temp)
        logits = self.classifier(feat)
        if return_embedding:
            embed = F.normalize(self.projector(feat), p=2, dim=1)
            return logits, embed
        return logits


def frequency_mask(mel, mask_width=3, n_masks_max=2):
    """Random frequency-band masking filled with per-sample global mean."""
    mel_masked = mel.clone()
    batch_size = mel.size(0)
    for i in range(batch_size):
        n_masks = torch.randint(1, n_masks_max + 1, (1,)).item()
        for _ in range(n_masks):
            w = torch.randint(mask_width // 2, mask_width + 1, (1,)).item()
            f0 = torch.randint(0, mel.size(2) - w, (1,)).item()
            mel_masked[i, :, f0 : f0 + w, :] = mel[i].mean()
    return mel_masked


def supcon_loss(features, labels, temperature=0.07):
    """Supervised contrastive loss on L2-normalized features."""
    n = features.size(0)
    sim = torch.matmul(features, features.T) / temperature
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    sim_masked = sim.clone()
    sim_masked[torch.eye(n, dtype=torch.bool, device=sim.device)] = float("-inf")
    log_sum_exp = torch.logsumexp(sim_masked, dim=1)

    n_pos = pos_mask.sum(dim=1)
    pos_sum = (sim * pos_mask).sum(dim=1)
    mean_pos = pos_sum / n_pos.clamp(min=1)
    loss = -mean_pos + log_sum_exp
    has_pos = n_pos > 0
    return loss[has_pos].mean()


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, criterion, device, lambda_sc=0.12, mask_width=3, n_masks_max=2):
    model.train()
    total_loss = 0.0
    for mel, temp, labels, _ in loader:
        mel = mel.to(device)
        temp = temp.to(device)
        labels = labels.to(device)
        batch_size = mel.size(0)

        mel_mask = frequency_mask(mel, mask_width, n_masks_max)
        mel_both = torch.cat([mel, mel_mask], dim=0)
        temp_both = torch.cat([temp, temp], dim=0)
        labels_both = torch.cat([labels, labels], dim=0)

        logits, embed = model(mel_both, temp_both, return_embedding=True)
        loss_cls = criterion(logits[:batch_size], labels)
        loss_supcon = supcon_loss(embed, labels_both, temperature=0.07)
        loss_main = loss_cls + lambda_sc * loss_supcon

        mel_mix, y_a, y_b, lam = mixup_data(mel, labels, alpha=0.2)
        logits_mix = model(mel_mix, temp)
        loss_mix = mixup_criterion(criterion, logits_mix, y_a, y_b, lam)

        loss = loss_main + loss_mix
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    val_loss = 0.0
    preds, targets = [], []

    for mel, temp, labels, _ in loader:
        mel = mel.to(device)
        temp = temp.to(device)
        labels = labels.to(device)
        outputs = model(mel, temp)
        val_loss += criterion(outputs, labels).item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    return val_loss / len(loader), acc, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to batches/ directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lambda_sc", type=float, default=0.12)
    parser.add_argument("--mask_width", type=int, default=3)
    parser.add_argument("--n_masks_max", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temp_norm",
        type=str,
        choices=["global", "local"],
        default="local",
        help="local: per-dataset min/max (~87%%); global: norm_stats temp_min/max",
    )
    parser.add_argument("--output_dir", type=str, default="./results_exp_e")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    folds = [
        {"name": "Fold 1 (15Hz test)", "train_speeds": [2, 3, 4], "test_speed": 1},
        {"name": "Fold 2 (30Hz test)", "train_speeds": [1, 3, 4], "test_speed": 2},
        {"name": "Fold 3 (45Hz test)", "train_speeds": [1, 2, 4], "test_speed": 3},
        {"name": "Fold 4 (60Hz test)", "train_speeds": [1, 2, 3], "test_speed": 4},
    ]

    results = []
    for fold in folds:
        print(f"\n{'=' * 60}")
        print(fold["name"])
        print(f"Train speeds: {fold['train_speeds']} -> Test speed: {fold['test_speed']} (load=0)")
        print(f"Temp norm: {args.temp_norm}")
        print("=" * 60)

        tn = args.temp_norm
        pool = AllWindowDataset(args.data_dir, fold["train_speeds"], load_id=0, split="full", temp_norm=tn)
        n_total = len(pool)
        perm = np.random.permutation(n_total)
        n_train = int(n_total * 0.8)

        train_ds = AllWindowDataset(
            args.data_dir,
            fold["train_speeds"],
            load_id=0,
            split="train",
            indices=perm[:n_train],
            temp_norm=tn,
        )
        val_ds = AllWindowDataset(
            args.data_dir,
            fold["train_speeds"],
            load_id=0,
            split="val",
            indices=perm[n_train:],
            temp_norm=tn,
        )
        test_ds = AllWindowDataset(
            args.data_dir, [fold["test_speed"]], load_id=0, split="full", temp_norm=tn
        )

        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = LightCNN_CBAM_SupCon(num_classes=8, in_channels=4).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        best_path = os.path.join(args.output_dir, f"best_fold{fold['test_speed']}.pth")

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                lambda_sc=args.lambda_sc,
                mask_width=args.mask_width,
                n_masks_max=args.n_masks_max,
            )
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch < 3:
                print(f"  Ep {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        print(f"\nBest val accuracy: {best_val_acc:.2f}%")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

        _, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
        print(f"TEST Accuracy: {test_acc:.2f}%")

        fault_names = [ID2FAULT[i] for i in range(8)]
        for cls_id in range(8):
            mask = np.array(test_labels) == cls_id
            if mask.sum() > 0:
                cls_acc = 100.0 * np.sum((np.array(test_preds) == cls_id) & mask) / mask.sum()
                print(f"  {ID2FAULT[cls_id]:5s}: {cls_acc:.1f}%")

        print("\n" + classification_report(test_labels, test_preds, target_names=fault_names, digits=3, zero_division=0))
        cm = confusion_matrix(test_labels, test_preds)
        print("  " + "".join(f"{n:>6s}" for n in fault_names))
        for i, name in enumerate(fault_names):
            print(f"{name:5s}" + "".join(f"{cm[i, j]:6d}" for j in range(8)))

        results.append(
            {
                "fold": fold["name"],
                "test_speed": fold["test_speed"],
                "test_acc": test_acc,
                "preds": test_preds,
                "labels": test_labels,
            }
        )

    print(f"\n{'=' * 60}")
    print("4-FOLD LOSO SUMMARY")
    print("=" * 60)
    accs = [r["test_acc"] for r in results]
    for r in results:
        print(f"  {r['fold']}: {r['test_acc']:.2f}%")
    print(f"  Average: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")

    np.savez(
        os.path.join(args.output_dir, "results.npz"),
        accs=np.array(accs),
        fold_names=[r["fold"] for r in results],
    )


if __name__ == "__main__":
    main()
