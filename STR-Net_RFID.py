import math
import os
import random
import numpy as np
import pywt
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from XRFDataset import HSWIRFDatasetNewMix
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
import torch.nn.functional as F
import logging
from opts import parse_opts



def set_random_seed(seed=1234):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set random seed before training
set_random_seed(1234)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# Core Network Modules
# =====================================================

class RFIDWaveletTransform(nn.Module):
    """
    Wavelet transform with learnable boundary handling
    """

    def __init__(self, wave='haar'):
        super().__init__()
        wavelet = pywt.Wavelet(wave)

        # Learnable boundary interpolation coefficient
        self.edge_scale = nn.Parameter(torch.ones(1) * 0.5)

        self.register_buffer('lo_filter', torch.tensor(wavelet.dec_lo[::-1]).float())
        self.register_buffer('hi_filter', torch.tensor(wavelet.dec_hi[::-1]).float())

        self.lo_filter = self.lo_filter.view(1, 1, -1)
        self.hi_filter = self.hi_filter.view(1, 1, -1)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, L]
        """
        B, C, L = x.shape

        # Adaptive boundary padding
        left_pad = x[:, :, 0:1] * self.edge_scale + x[:, :, 1:2] * (1 - self.edge_scale)
        right_pad = x[:, :, -1:] * self.edge_scale + x[:, :, -2:-1] * (1 - self.edge_scale)
        x_pad = torch.cat([left_pad, x, right_pad], dim=2)

        lo = F.conv1d(
            x_pad,
            self.lo_filter.repeat(C, 1, 1),
            stride=2,
            groups=C
        )
        hi = F.conv1d(
            x_pad,
            self.hi_filter.repeat(C, 1, 1),
            stride=2,
            groups=C
        )

        return torch.cat([lo, hi], dim=1)


class RFIDTemporalBoost(nn.Module):
    """
    Temporal-domain feature enhancement module
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv1d(channels // 4, channels, 1)
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return x + self.gate * self.conv(x)


class HyperDynamicRouter_RFID(nn.Module):
    """
    Hyper-dynamic routing module for RFID signals

    Improvements:
    1) Safe automatic group selection
    2) Pulse gate uses Avg + Max pooling
    3) Routing features use mean/std + positional mean
    4) Residual strength controlled by learnable gamma
    5) Optional routing weight output for analysis
    """

    def __init__(self, in_channels, base_groups=8, hidden=32, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.temperature = temperature

        # Safe group selection
        g = min(base_groups, in_channels)
        while in_channels % g != 0 and g > 1:
            g -= 1

        # Spatial branch
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        # Pulse gate branch
        self.pulse_gate = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        # Lightweight positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        # Router network
        h = min(hidden, max(8, in_channels // 2))
        self.router = nn.Sequential(
            nn.Linear(in_channels * 3, h),
            nn.GELU(),
            nn.Linear(h, 2)
        )

        # Residual scaling factor
        self.gamma = nn.Parameter(torch.tensor(0.0))

        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_weights=False):
        """
        x: [B, C, L]
        """
        B, C, L = x.shape

        spatial = self.spatial(x)

        avg = x.mean(dim=2, keepdim=True)
        mx = x.amax(dim=2, keepdim=True)
        gate = self.pulse_gate(torch.cat([avg, mx], dim=1))
        pulse = x * gate

        pos_feat = self.pos_encoder(x)

        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=False)
        posm = pos_feat.mean(dim=2)

        route_feat = torch.cat([mean, std, posm], dim=1)
        logits = self.router(route_feat)

        weights = F.softmax(logits / max(self.temperature, 1e-6), dim=1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1).mean()

        fused = (
            weights[:, 0].view(B, 1, 1) * spatial +
            weights[:, 1].view(B, 1, 1) * pulse
        )

        out = x + self.gamma * fused

        if return_weights:
            return out, entropy, weights
        return out, entropy


class HyperdynamicTimeFrequencyRoutingAttention(nn.Module):
    """
    Joint time-frequency attention module using wavelet decomposition
    """

    def __init__(self, channels):
        super().__init__()
        self.dwt = RFIDWaveletTransform('haar')

        self.res_conv = nn.Conv1d(channels, 2 * channels, 1)
        self.freq_proj = nn.Conv1d(2 * channels, channels, 3, padding=1)

        self.time_branch = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=8),
            RFIDTemporalBoost(channels),
            nn.GELU()
        )

        self.fuse_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        time_feat = self.time_branch(x)

        if x.size(2) >= 4:
            dwt_feat = self.dwt(x)
            C = x.size(1)

            lo_residual = dwt_feat[:, :C, :]
            residual = self.res_conv(lo_residual)

            dwt_feat = dwt_feat + residual

            freq_feat = F.interpolate(
                self.freq_proj(dwt_feat),
                size=x.size(2),
                mode='linear',
                align_corners=True
            )

            return time_feat + self.fuse_weight.sigmoid() * freq_feat

        return time_feat


# =====================================================
# Backbone Network
# =====================================================

class BasicBlock1D_RFID(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

        self.attention = HyperdynamicTimeFrequencyRoutingAttention(out_channels)
        self.router = HyperDynamicRouter_RFID(out_channels)

    def forward(self, x):
        identity = x

        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = self.attention(x)
        x, entropy = self.router(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = F.gelu(x + identity)
        return x, entropy


class ResNet1D_RFID(nn.Module):
    def __init__(self, input_channels=23, num_classes=55):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        self.in_channels = 32
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_c:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

        layers = [BasicBlock1D_RFID(self.in_channels, out_c, stride, downsample)]
        self.in_channels = out_c

        for _ in range(1, blocks):
            layers.append(BasicBlock1D_RFID(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        for blk in self.layer1:
            x, _ = blk(x)
        for blk in self.layer2:
            x, _ = blk(x)
        for blk in self.layer3:
            x, _ = blk(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =====================================================
# Training & Evaluation
# =====================================================

def SpectralTemporalRepresentationNetwork(input_channels=23, num_classes=55):
    return ResNet1D_RFID(input_channels, num_classes)


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0

    with torch.no_grad():
        for _, rfid_data, labels in tqdm(data_loader, leave=False):
            rfid_data, labels = rfid_data.to(device), labels.to(device)

            logits = model(rfid_data)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=1
    )
    acc = accuracy_score(all_labels, all_preds)

    return total_loss / len(data_loader), precision, recall, f1, acc, all_labels, all_preds


def plot_confusion_matrix(labels, preds, filename):
    cm = confusion_matrix(labels, preds)
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("RFID Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join("results", filename))
    plt.close()


def main():
    args = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HSWIRFDatasetNewMix(is_train=True, scene="dml")
    val_dataset = HSWIRFDatasetNewMix(is_train=False, scene="dml")

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = SpectralTemporalRepresentationNetwork(
        input_channels=24,
        num_classes=18
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    epochs = args.epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=epochs * len(train_loader)
    )

    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for _, rfid_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            rfid_data, labels = rfid_data.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(rfid_data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        val_loss, p, r, f1, acc, labels_all, preds_all = evaluate_model(
            model, val_loader, device, criterion
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model-zy.pth")
            plot_confusion_matrix(labels_all, preds_all, "rfid_confusion_matrix.png")

        logger.info(
            f"Epoch {epoch+1}: "
            f"TrainLoss={total_loss/len(train_loader):.4f} "
            f"ValLoss={val_loss:.4f} "
            f"Acc={acc:.4f} "
            f"F1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
