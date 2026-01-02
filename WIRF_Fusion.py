import os
import random
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging

from XRFDataset import HSWIRFDatasetNewMix
from opts import parse_opts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Utils
# =========================
def set_random_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(1234)


def plot_confusion_matrix(labels, preds, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not os.path.exists('results'):
        os.makedirs('results')
    save_path = os.path.join('results', filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved at: {save_path}")


def evaluate_model_fusion(model, data_loader, device, criterion):
    model.eval()
    total_val_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for wifi_data, rfid_data, labels in tqdm(data_loader, desc="Evaluating Fusion", leave=False):
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels = labels.to(device)

            logits = model(wifi_data, rfid_data)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=1
    )
    acc = accuracy_score(all_labels, all_preds)

    return total_val_loss / max(1, len(data_loader)), precision, recall, f1, acc, all_labels, all_preds


# =========================
# WiFi Branch (your TFDA Ultra)
# =========================
class HyperDynamicRouter(nn.Module):
    def __init__(self, in_channels, num_paths=3, expansion=4, attn_hidden=32, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_paths = num_paths
        self.temperature = temperature

        g = max(1, in_channels // 4)
        while in_channels % g != 0 and g > 1:
            g -= 1

        self.path0 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        self.path1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, attn_hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv1d(attn_hidden, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

        mid = max(1, in_channels // expansion)
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
        )

        self.router = nn.Sequential(
            nn.Linear(in_channels * 2, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, num_paths)
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
        )

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, L = x.shape

        f0 = self.path0(x)
        w1 = self.path1(x)
        f1 = x * w1
        f2 = self.path2(x)
        feats = [f0, f1, f2]

        g_mean = x.mean(dim=2)
        g_max = x.max(dim=2).values
        logits = self.router(torch.cat([g_mean, g_max], dim=1))

        route_weights = F.softmax(logits / max(self.temperature, 1e-6), dim=1)

        fused = 0.0
        for i in range(self.num_paths):
            fused = fused + route_weights[:, i].view(B, 1, 1) * feats[i]

        fused = self.fuse(fused)
        out = x + self.gamma * fused
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
        self.depthwise = nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels)
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
        weights = self.weights(combined)  # [B,2,?]
        return weights[:, 0:1] * time_feat + weights[:, 1:2] * freq_feat


class SharedFrequencyProcessor(nn.Module):
    def __init__(self, in_channels, reduction, levels=2):
        super().__init__()
        self.levels = levels
        self.band_channels = in_channels // levels
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.band_channels, max(1, self.band_channels // reduction), 3,
                          padding=1, groups=1),
                nn.BatchNorm1d(max(1, self.band_channels // reduction)),
                nn.GELU(),
                nn.Conv1d(max(1, self.band_channels // reduction), self.band_channels, 1),
                nn.Sigmoid()
            ) for _ in range(levels)
        ])

        self.cross_band = nn.Conv1d(in_channels, in_channels // 2, 3, padding=1, groups=1)

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


class FastTFDA_SEBlock1D_Ultra(nn.Module):
    def __init__(self, channels, reduction=4, J=1):
        super().__init__()
        self.J = min(J, 2)

        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.time_branch = nn.Sequential(
            DepthwiseSeparableConv(channels, reduction),
            HyperDynamicRouter(channels // reduction),
            nn.Sequential(
                nn.Conv1d(channels // reduction, channels, 1),
                nn.BatchNorm1d(channels),
                nn.GELU()
            )
        )

        self.dwt = FastWaveletTransform('coif3')
        self.freq_processor = SharedFrequencyProcessor(in_channels=2 * channels, reduction=reduction, levels=self.J + 1)
        self.fusion_gate = FusionGate(channels)

    def forward(self, x):
        gate = self.gate_net(x)

        with torch.amp.autocast('cuda', enabled=False):
            time_feat = self.time_branch(x * gate)
            dwt_out = self.dwt(x)
            freq_feat = self.freq_processor(dwt_out)
            freq_feat = F.interpolate(freq_feat, size=x.size(2), mode='nearest')
        return self.fusion_gate(time_feat, freq_feat)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.attention = FastTFDA_SEBlock1D_Ultra(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet1D_WiFi(nn.Module):
    def __init__(self, block, layers, input_channels=270, num_classes=18):
        super().__init__()
        self.in_channels = 64
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
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)  # [B,512]
        if return_feat:
            return feat
        return self.fc(feat)


def resnet18_1d_enhanced_wifi(input_channels=270, num_classes=18):
    return ResNet1D_WiFi(BasicBlock1D, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes)


# =========================
# RFID Branch (your RFID net)
# =========================
class RFIDWaveletTransform(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        wavelet = pywt.Wavelet(wave)
        self.edge_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer('lo_filter', torch.tensor(wavelet.dec_lo[::-1]).float())
        self.register_buffer('hi_filter', torch.tensor(wavelet.dec_hi[::-1]).float())
        self.lo_filter = self.lo_filter.view(1, 1, -1)
        self.hi_filter = self.hi_filter.view(1, 1, -1)

    def forward(self, x):
        B, C, L = x.shape
        left_pad = x[:, :, 0:1] * self.edge_scale + x[:, :, 1:2] * (1 - self.edge_scale)
        right_pad = x[:, :, -1:] * self.edge_scale + x[:, :, -2:-1] * (1 - self.edge_scale)
        x_pad = torch.cat([left_pad, x, right_pad], dim=2)
        lo = F.conv1d(x_pad, self.lo_filter.repeat(C, 1, 1), stride=2, groups=C)
        hi = F.conv1d(x_pad, self.hi_filter.repeat(C, 1, 1), stride=2, groups=C)
        return torch.cat([lo, hi], dim=1)


class RFIDTemporalBoost(nn.Module):
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
    def __init__(self, in_channels, base_groups=8, hidden=32, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        g = min(base_groups, in_channels)
        while in_channels % g != 0 and g > 1:
            g -= 1

        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        self.pulse_gate = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.pos_encoder = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        h = min(hidden, max(8, in_channels // 2))
        self.router = nn.Sequential(
            nn.Linear(in_channels * 3, h),
            nn.GELU(),
            nn.Linear(h, 2)
        )

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
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
        w = F.softmax(logits / max(self.temperature, 1e-6), dim=1)

        fused = w[:, 0].view(B, 1, 1) * spatial + w[:, 1].view(B, 1, 1) * pulse
        out = x + self.gamma * fused
        return out


class FastTFDA_SEBlock1D_RFID(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = RFIDWaveletTransform('haar')
        self.res_conv = nn.Conv1d(channels, 2 * channels, 1)
        self.freq_proj = nn.Conv1d(2 * channels, channels, 3, padding=1)

        g = min(8, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.time_branch = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=g),
            RFIDTemporalBoost(channels),
            nn.GELU()
        )
        self.fuse_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        time_feat = self.time_branch(x)
        if x.size(2) < 4:
            return time_feat

        dwt_feat = self.dwt(x)  # [B,2C,L/2]
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


class BasicBlock1D_RFID(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.attention = FastTFDA_SEBlock1D_RFID(out_channels)
        self.router = HyperDynamicRouter_RFID(out_channels)

    def forward(self, x):
        identity = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = self.attention(x)
        x = self.router(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.gelu(x + identity)
        return out


class ResNet1D_RFID(nn.Module):
    def __init__(self, input_channels=24, num_classes=18):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.in_channels = 32
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_c, blocks, stride=1):
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

    def forward(self, x, return_feat=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)  # [B,256]
        if return_feat:
            return feat
        return self.fc(feat)


def resnet18_rfid(input_channels=24, num_classes=18):
    return ResNet1D_RFID(input_channels=input_channels, num_classes=num_classes)


# =========================
# Fusion Model (concat)
# =========================
class FusionNet(nn.Module):
    def __init__(self, wifi_in_ch=270, rfid_in_ch=24, num_classes=18, fusion_hidden=256, dropout=0.2):
        super().__init__()
        self.wifi = resnet18_1d_enhanced_wifi(input_channels=wifi_in_ch, num_classes=num_classes)
        self.rfid = resnet18_rfid(input_channels=rfid_in_ch, num_classes=num_classes)

        wifi_feat_dim = 512
        rfid_feat_dim = 256
        fused_dim = wifi_feat_dim + rfid_feat_dim

        self.rfid_weight = 1.3  # RFID特征权重
        self.wifi_weight = 1.0  # WiFi特征权重

        self.fuser = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, wifi_x, rfid_x):
        wifi_feat = self.wifi(wifi_x, return_feat=True)
        rfid_feat = self.rfid(rfid_x, return_feat=True)

        # 应用权重
        weighted_wifi_feat = wifi_feat * self.wifi_weight
        weighted_rfid_feat = rfid_feat * self.rfid_weight

        fused = torch.cat([weighted_wifi_feat, weighted_rfid_feat], dim=1)
        fused = self.fuser(fused)
        logits = self.classifier(fused)
        return logits


# =========================
# Main
# =========================
def main():
    args = parse_opts()

    gpu = getattr(args, "gpu", 0)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = HSWIRFDatasetNewMix(is_train=True, scene="dml")
    val_dataset = HSWIRFDatasetNewMix(is_train=False, scene="dml")

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False
    )

    wifi_in_ch = 270
    rfid_in_ch = 24
    num_classes = 18
    epochs = getattr(args, "epoch", 200)

    model = FusionNet(
        wifi_in_ch=wifi_in_ch,
        rfid_in_ch=rfid_in_ch,
        num_classes=num_classes,
        fusion_hidden=256,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=epochs * len(train_loader),
        pct_start=0.3
    )

    # ---- Track ONLY the best epoch ----
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

        for wifi_data, rfid_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(wifi_data, rfid_data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        val_loss, precision, recall, f1, acc, all_labels, all_preds = evaluate_model_fusion(
            model, val_loader, device, criterion
        )

        # ---- Only update best record (no per-epoch printing) ----
        if f1 > best["f1"]:
            best.update({
                "epoch": epoch + 1,
                "f1": float(f1),
                "acc": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "val_loss": float(val_loss),
                "train_loss": float(train_loss),
                "labels": all_labels,
                "preds": all_preds,
            })
            torch.save(model.state_dict(), "best_fusion_wifi_rfid.pth")

    # ---- Print ONLY the best epoch metrics ----
    logger.info(
        f"[BEST] Epoch={best['epoch']} "
        f"TrainLoss={best['train_loss']:.4f} "
        f"ValLoss={best['val_loss']:.4f} "
        f"Acc={best['acc']:.4f} "
        f"P={best['precision']:.4f} "
        f"R={best['recall']:.4f} "
        f"F1={best['f1']:.4f}"
    )

    # Optional: save confusion matrix for the best epoch
    if best["labels"] is not None:
        plot_confusion_matrix(
            best["labels"],
            best["preds"],
            title=f"Fusion Confusion Matrix (Best Epoch {best['epoch']})",
            filename="fusion_confusion_best.png"
        )


if __name__ == "__main__":
    main()
