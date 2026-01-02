import os
import random
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from XRFDataset import HSWIRFDatasetNewMix
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
import logging
# from XRFDataset import XRFBertDatasetNewMix
from opts import parse_opts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def set_random_seed(seed=1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Call before main() for deterministic behavior
set_random_seed(1234)


class HyperDynamicRouter(nn.Module):
    """
    Improvements:
    1) Use route_weights for true weighted fusion
    2) Add temperature + entropy (can be used as regularization)
    3) Safe group selection to avoid groups=0 / non-divisible cases
    4) Use lightweight fuse (1x1 + DWConv) instead of concat+dim_adapter (more efficient)
    """
    def __init__(self, in_channels, num_paths=3, expansion=4, attn_hidden=32, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_paths = num_paths
        self.temperature = temperature

        # Safe group selection: at least 1 and must divide in_channels
        g = max(1, in_channels // 4)
        while in_channels % g != 0 and g > 1:
            g -= 1

        # Path0: grouped/depthwise-like conv (more stable)
        self.path0 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        # Path1: channel attention (Conv1d version to avoid Flatten/Unflatten)
        self.path1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, attn_hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv1d(attn_hidden, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

        # Path2: dynamic sparse bottleneck
        mid = max(1, in_channels // expansion)
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
        )

        # Router: global statistics (mean/max) for better discrimination
        self.router = nn.Sequential(
            nn.Linear(in_channels * 2, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, num_paths)
        )

        # Lightweight fusion refinement
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
        )

        # Residual scaling (init to 0 for stability)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, return_entropy=False):
        B, C, L = x.shape

        f0 = self.path0(x)                            # [B,C,L]
        w1 = self.path1(x)                            # [B,C,1]
        f1 = x * w1                                   # [B,C,L]
        f2 = self.path2(x)                            # [B,C,L]

        feats = [f0, f1, f2]

        # Route logits: concat(mean, max) -> [B, 2C]
        g_mean = x.mean(dim=2)
        g_max = x.max(dim=2).values
        logits = self.router(torch.cat([g_mean, g_max], dim=1))  # [B,P]

        # Temperature-softmax
        route_weights = F.softmax(logits / max(self.temperature, 1e-6), dim=1)  # [B,P]

        fused = 0.0
        for i in range(self.num_paths):
            fused = fused + route_weights[:, i].view(B, 1, 1) * feats[i]

        fused = self.fuse(fused)

        out = x + self.gamma * fused

        if return_entropy:
            entropy = -torch.sum(route_weights * torch.log(route_weights + 1e-10), dim=1).mean()
            return out, entropy, route_weights
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        effective_reduction = max(2, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // effective_reduction),
            nn.GELU(),
            nn.Linear(channel // effective_reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        attn = self.fc(self.gap(x).view(b, c))
        return x * attn.unsqueeze(-1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 3,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm1d(in_channels // reduction),
            nn.GELU(),
            ChannelAttention(in_channels // reduction)
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FusionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv1d(2 * channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, time_feat, freq_feat):
        combined = torch.cat([time_feat, freq_feat], dim=1)
        weights = self.weights(combined)  # [B,2,C]
        return weights[:, 0:1] * time_feat + weights[:, 1:2] * freq_feat


class FastTFDA_SEBlock1D_Ultra(nn.Module):
    def __init__(self, channels, reduction=4, J=1):
        super().__init__()
        self.J = min(J, 2)
        self.channels = channels

        # Gating network with corrected channel reduction
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # Ensure correct channel flow through the time branch
        self.time_branch = nn.Sequential(
            DepthwiseSeparableConv(channels, reduction),
            HyperDynamicRouter(channels // reduction),
            nn.Sequential(
                nn.Conv1d(channels // reduction, channels, 1),
                nn.BatchNorm1d(channels),
                nn.GELU()
            )
        )

        # Frequency-domain processing
        self.dwt = FastWaveletTransform('coif3')
        self.freq_processor = SharedFrequencyProcessor(
            in_channels=2 * channels,
            reduction=reduction,
            levels=self.J + 1
        )
        self.fusion_gate = FusionGate(channels)

    def forward(self, x):
        gate = self.gate_net(x)

        # Disable AMP inside this block for numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            time_feat = self.time_branch(x * gate)

            dwt_out = self.dwt(x)
            freq_feat = self.freq_processor(dwt_out)
            freq_feat = F.interpolate(freq_feat, size=x.size(2), mode='nearest')
            freq_feat = self.channel_adjust(freq_feat)

        return self.fusion_gate(time_feat, freq_feat)

    def channel_adjust(self, x):
        # Placeholder for optional channel alignment
        return x


class SharedFrequencyProcessor(nn.Module):
    def __init__(self, in_channels, reduction, levels=2):
        super().__init__()
        self.levels = levels
        self.band_channels = in_channels // levels
        self.attention_weights = nn.Parameter(torch.ones(levels))

        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.band_channels, self.band_channels // reduction, 3,
                          padding=1, groups=max(1, self.band_channels // reduction // 4)),
                nn.BatchNorm1d(self.band_channels // reduction),
                nn.GELU(),
                nn.Conv1d(self.band_channels // reduction, self.band_channels, 1),
                nn.Sigmoid()
            ) for _ in range(levels)
        ])

        self.cross_band = nn.Conv1d(in_channels, in_channels // 2, 3,
                                    padding=1, groups=8)

    def forward(self, freq):
        B, C, L = freq.shape
        outputs = []

        for i in range(self.levels):
            band = freq[:, i * self.band_channels: (i + 1) * self.band_channels]
            resized = F.interpolate(band, scale_factor=2 ** i, mode='nearest')
            processed = self.processors[i](resized)
            processed = F.interpolate(processed, size=L, mode='nearest')
            outputs.append(processed)

        fused = torch.cat(outputs, dim=1)
        return self.cross_band(fused)


class FastWaveletTransform(nn.Module):
    def __init__(self, wave='coif3'):
        super().__init__()
        wavelet = pywt.Wavelet(wave)
        self.register_buffer('lo_filter', torch.tensor(wavelet.dec_lo[::-1]).float())
        self.register_buffer('hi_filter', torch.tensor(wavelet.dec_hi[::-1]).float())
        self.lo_filter = self.lo_filter.view(1, 1, -1)
        self.hi_filter = self.hi_filter.view(1, 1, -1)

    def forward(self, x):
        B, C, L = x.shape
        x_pad = F.pad(x, (self.lo_filter.size(-1) // 2, 0), mode='replicate')

        lo = F.conv1d(x_pad, self.lo_filter.repeat(C, 1, 1), stride=2, groups=C)
        hi = F.conv1d(x_pad, self.hi_filter.repeat(C, 1, 1), stride=2, groups=C)
        return torch.cat([lo, hi], dim=1)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, attention_type='tfda'):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

        # Select attention module
        if attention_type == 'tfda':
            self.attention = FastTFDA_SEBlock1D_Ultra(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ResNet1D with Enhanced Stem
class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_channels=270, num_classes=55):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # EfficientNet-style enhanced stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Helper function to create ResNet18-1D
def resnet18_1d_enhanced(input_channels=270, num_classes=10):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes)


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    total_val_loss = 0
    all_labels = []
    all_preds = []
    all_wifi_preds = []

    with torch.no_grad():
        for wifi_data, rfid_data, labels in tqdm(data_loader, desc=f"Evaluating WiFi Classifier", leave=False):
            wifi_data, labels = wifi_data.to(device), labels.to(device)

            # Forward pass for WiFi classifier
            wifi_output = model(wifi_data)

            val_loss = criterion(wifi_output, labels)
            total_val_loss += val_loss.item()

            _, wifi_predicted = torch.max(wifi_output.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(wifi_predicted.cpu().numpy())
            all_wifi_preds.extend(wifi_predicted.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=1
    )
    accuracy = accuracy_score(all_labels, all_preds)

    logger.info(
        f"WiFi Classifier - Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return total_val_loss / len(data_loader), precision, recall, f1, accuracy, all_labels, all_preds


def compute_supervised_loss(wifi_output, labels, criterion):
    # Cross-entropy loss for WiFi classifier
    wifi_loss = criterion(wifi_output, labels)

    # Total loss equals WiFi loss (single-modality training)
    total_loss = wifi_loss
    return total_loss


def plot_confusion_matrix(labels, preds, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Ensure the save directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    save_path = os.path.join('results', filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")


def main():
    model_name = "train_dml"
    logger.info(f"Starting training for model: {model_name}")
    args = parse_opts()

    # Device selection
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # WiFi-only dataset (DataLoader still yields wifi_data, rfid_data, labels)
    train_dataset = HSWIRFDatasetNewMix(is_train=True, scene="dml")
    val_dataset = HSWIRFDatasetNewMix(is_train=False, scene="dml")

    train_data = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False
    )
    val_data = Data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False
    )

    # Model
    model = resnet18_1d_enhanced(input_channels=270, num_classes=18).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # Use args.epoch if provided; otherwise default to 200
    epochs = int(getattr(args, "epoch", 200))

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=epochs * len(train_data),
        pct_start=0.3
    )

    criterion = nn.CrossEntropyLoss()

    # Track ONLY the best epoch metrics
    best = {
        "epoch": -1,
        "f1": -1.0,
        "acc": None,
        "precision": None,
        "recall": None,
        "val_loss": None,
        "train_loss": None,
        "labels": None,
        "preds": None,
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for wifi_data, rfid_data, labels in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", leave=False):
            wifi_data = wifi_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(wifi_data)

            # Supervised loss (keep your original function)
            loss = compute_supervised_loss(logits, labels, criterion)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_data))

        # Validation (no per-epoch printing other than logger inside evaluate_model)
        val_loss, precision, recall, f1, accuracy, all_labels, all_preds = evaluate_model(
            model, val_data, device, criterion
        )

        # Update best checkpoint based on macro-F1
        if f1 > best["f1"]:
            best.update({
                "epoch": epoch + 1,
                "f1": float(f1),
                "acc": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "val_loss": float(val_loss),
                "train_loss": float(train_loss),
                "labels": all_labels,
                "preds": all_preds,
            })
            torch.save(model.state_dict(), "best_model-zy.pth")

    # Print ONLY the best epoch
    logger.info(
        f"[BEST] Epoch={best['epoch']}/{epochs} "
        f"TrainLoss={best['train_loss']:.4f} "
        f"ValLoss={best['val_loss']:.4f} "
        f"Acc={best['acc']:.4f} "
        f"P={best['precision']:.4f} "
        f"R={best['recall']:.4f} "
        f"F1={best['f1']:.4f}"
    )

    # Confusion matrix for the best epoch
    if best["labels"] is not None:
        plot_confusion_matrix(
            best["labels"],
            best["preds"],
            title=f"Confusion Matrix (Best Epoch {best['epoch']})",
            filename="confusion_matrix.png"
        )

        # Per-class accuracy for the best epoch
        all_labels_np = np.array(best["labels"])
        all_preds_np = np.array(best["preds"])
        num_classes = int(np.max(all_labels_np)) + 1

        logger.info("Per-class accuracy for best epoch:")
        for cls in range(num_classes):
            cls_mask = (all_labels_np == cls)
            total = int(np.sum(cls_mask))
            correct = int(np.sum(all_preds_np[cls_mask] == cls)) if total > 0 else 0
            acc_cls = (correct / total) if total > 0 else 0.0
            logger.info(f"  Class {cls:02d}: Accuracy = {acc_cls:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
