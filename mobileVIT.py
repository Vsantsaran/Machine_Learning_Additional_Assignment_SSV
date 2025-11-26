#!/usr/bin/env python3
"""
Cotton Leaf Disease Detection Using Optimized MobileViT
End-to-End Training Pipeline with Multi-GPU Support

Author: Research Implementation
Dataset: SAR-CLD-2024 Cotton Leaf Disease Dataset
Architecture: Optimized MobileViT with Multi-Scale Attention
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)

# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

DISEASE_CLASSES = [
    "Bacterial_Blight",
    "Curl_Virus",
    "Healthy_Leaf",
    "Herbicide_Growth_Damage",
    "Leaf_Hopper_Jassids",
    "Leaf_Redding",
    "Leaf_Variegation"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str, rank: int = 0):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)

    if rank == 0:
        # Output log
        out_log = os.path.join(log_dir, 'out.log')
        err_log = os.path.join(log_dir, 'error.log')

        # Create formatters
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for output
        file_handler = logging.FileHandler(out_log)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # File handler for errors
        error_handler = logging.FileHandler(err_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        return logger
    else:
        # For non-master processes, suppress logging
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        return logger


# ============================================================================
# DATASET CLASS
# ============================================================================

class CottonLeafDataset(Dataset):
    """Cotton Leaf Disease Dataset with augmentation support"""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (256, 256), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset_paths(data_root: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """Load all image paths and labels from both original and augmented datasets"""
    all_paths = []
    all_labels = []

    aug_dir = os.path.join(data_root, 'aug_ds')
    orig_dir = os.path.join(data_root, 'orig_ds')

    for class_idx, class_name in enumerate(class_names):
        # Load from augmented dataset
        aug_class_dir = os.path.join(aug_dir, class_name)
        if os.path.exists(aug_class_dir):
            for img_file in os.listdir(aug_class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_paths.append(os.path.join(aug_class_dir, img_file))
                    all_labels.append(class_idx)

        # Load from original dataset
        orig_class_dir = os.path.join(orig_dir, class_name)
        if os.path.exists(orig_class_dir):
            for img_file in os.listdir(orig_class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_paths.append(os.path.join(orig_class_dir, img_file))
                    all_labels.append(class_idx)

    return all_paths, all_labels


def create_data_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List, List, List, List, List, List]:
    """Create stratified train/val/test splits"""

    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_val_labels
    )

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class RandAugment:
    """RandAugment implementation"""
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        self.augment_list = [
            (transforms.AutoAugment(), 0.5),
            (transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), 0.5),
            (transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 0.5),
            (transforms.RandomPerspective(distortion_scale=0.2, p=0.5), 0.5),
        ]

    def __call__(self, img):
        ops = random.sample(self.augment_list, self.n)
        for op, prob in ops:
            if random.random() < prob:
                img = op(img)
        return img


class MixUp:
    """MixUp augmentation"""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix augmentation"""
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if random.random() > self.prob:
            return batch_x, batch_y, batch_y, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        _, _, h, w = batch_x.size()
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        y_a, y_b = batch_y, batch_y[index]
        return batch_x, y_a, y_b, lam


def get_transforms(image_size: int = 256, is_training: bool = True) -> transforms.Compose:
    """Get data transforms for training/validation"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandAugment(n=2, m=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


# ============================================================================
# MOBILEVIT ARCHITECTURE
# ============================================================================

class ConvBNActivation(nn.Module):
    """Convolution + BatchNorm + Activation"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: nn.Module = nn.SiLU
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2 style)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4
    ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.append(ConvBNActivation(in_channels, hidden_dim, kernel_size=1))

        # Depthwise
        layers.extend([
            ConvBNActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            # Projection
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MultiScaleAttention(nn.Module):
    """Multi-Scale Multi-Head Attention with stability improvements"""
    def __init__(
        self, 
        dim: int,
        num_heads: int = 4,
        patch_sizes: List[int] = [2, 4, 8],
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections for each scale
        self.qkv_projections = nn.ModuleList([
            nn.Linear(dim, dim * 3) for _ in patch_sizes
        ])
        
        self.proj = nn.Linear(dim * len(patch_sizes), dim)
        self.dropout = nn.Dropout(dropout)
        
        # Add layer normalization for stability
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Normalize input for stability
        x = self.norm(x)
        
        outputs = []
        
        for idx, patch_size in enumerate(self.patch_sizes):
            # Apply Q, K, V projection
            qkv = self.qkv_projections[idx](x).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Scaled dot-product attention with clipping
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # Clip attention scores to prevent overflow
            attn = torch.clamp(attn, min=-10, max=10)
            
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            outputs.append(out)
        
        # Concatenate multi-scale outputs
        x = torch.cat(outputs, dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Scale Attention and GELU"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        patch_sizes: List[int] = [2, 4, 8],
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(dim, num_heads, patch_sizes, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # Using GELU as specified
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT Block - Simplified and Robust Implementation
    This version uses a simpler patch embedding strategy that's more reliable
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        num_heads: int = 4,
        transformer_depth: int = 2,
        patch_size: int = 2,
        patch_sizes: List[int] = [2, 4, 8],
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_h = patch_size
        self.patch_w = patch_size
        
        # Local representation (keeps spatial structure)
        self.conv_kxk = ConvBNActivation(in_channels, in_channels, kernel_size=3)
        
        # Project to transformer dimension
        self.conv_1x1_in = ConvBNActivation(in_channels, transformer_dim, kernel_size=1)
        
        # Transformer blocks for global representation
        self.transformers = nn.ModuleList([
            TransformerBlock(
                transformer_dim, num_heads, mlp_ratio, 
                patch_sizes, dropout
            ) for _ in range(transformer_depth)
        ])
        
        # Project back from transformer dimension
        self.conv_1x1_out = ConvBNActivation(transformer_dim, in_channels, kernel_size=1)
        
        # Final projection
        self.conv_proj = ConvBNActivation(2 * in_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Save input for skip connection
        res = x
        
        # Local representation
        x = self.conv_kxk(x)
        
        # Project to transformer dimension
        x = self.conv_1x1_in(x)  # (B, transformer_dim, H, W)
        
        # Get dimensions
        _, C_t, H_t, W_t = x.shape
        
        # Calculate number of patches
        n_h = H_t // self.patch_h
        n_w = W_t // self.patch_w
        
        # If dimensions not divisible, use interpolation
        if H_t % self.patch_h != 0 or W_t % self.patch_w != 0:
            new_h = n_h * self.patch_h
            new_w = n_w * self.patch_w
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            H_t, W_t = new_h, new_w
            n_h = H_t // self.patch_h
            n_w = W_t // self.patch_w
        
        # Reshape for transformer: (B, C, H, W) -> (B, C, n_h, patch_h, n_w, patch_w)
        x = x.reshape(B, C_t, n_h, self.patch_h, n_w, self.patch_w)
        
        # Rearrange: (B, C, n_h, patch_h, n_w, patch_w) -> (B, n_h, n_w, patch_h, patch_w, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        
        # Flatten spatial dimensions within patches: (B, n_h, n_w, patch_h*patch_w, C)
        x = x.reshape(B, n_h * n_w, self.patch_h * self.patch_w, C_t)
        
        # Average pool within each patch to get: (B, num_patches, C)
        x = x.mean(dim=2)  # (B, n_h * n_w, C_t)
        
        # Apply transformer blocks
        for transformer in self.transformers:
            x = transformer(x)
        
        # Reshape back to image: (B, num_patches, C) -> (B, n_h, n_w, C)
        x = x.reshape(B, n_h, n_w, C_t)
        
        # Expand patches: (B, n_h, n_w, C) -> (B, n_h, n_w, patch_h, patch_w, C)
        x = x.unsqueeze(3).unsqueeze(4)
        x = x.expand(-1, -1, -1, self.patch_h, self.patch_w, -1)
        
        # Rearrange back: (B, n_h, n_w, patch_h, patch_w, C) -> (B, C, n_h, patch_h, n_w, patch_w)
        x = x.permute(0, 5, 1, 3, 2, 4)
        
        # Merge patches: (B, C, H, W)
        x = x.reshape(B, C_t, H_t, W_t)
        
        # Interpolate back to original size if needed
        if x.shape[2] != res.shape[2] or x.shape[3] != res.shape[3]:
            x = F.interpolate(x, size=(res.shape[2], res.shape[3]), mode='bilinear', align_corners=False)
        
        # Project back
        x = self.conv_1x1_out(x)  # (B, in_channels, H, W)
        
        # Concatenate with input and project
        x = torch.cat([res, x], dim=1)  # (B, 2*in_channels, H, W)
        x = self.conv_proj(x)  # (B, out_channels, H, W)
        
        return x


class OptimizedMobileViT(nn.Module):
    """Optimized MobileViT for Cotton Leaf Disease Detection"""
    def __init__(
        self,
        num_classes: int = 7,
        input_size: int = 256,
        width_multiplier: float = 1.2,
        expansion_ratio: int = 4,
        transformer_depth: int = 2,
        num_heads: int = 4,
        patch_size: int = 2,
        patch_sizes: List[int] = [2, 4, 8],
        dropout: float = 0.2,
        stochastic_depth: float = 0.1
    ):
        super().__init__()

        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Stem
        self.stem = nn.Sequential(
            ConvBNActivation(3, make_divisible(32 * width_multiplier), kernel_size=3, stride=2),
            InvertedResidual(
                make_divisible(32 * width_multiplier),
                make_divisible(64 * width_multiplier),
                stride=1,
                expand_ratio=expansion_ratio
            )
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            InvertedResidual(
                make_divisible(64 * width_multiplier),
                make_divisible(64 * width_multiplier),
                stride=2,
                expand_ratio=expansion_ratio
            ),
            InvertedResidual(
                make_divisible(64 * width_multiplier),
                make_divisible(64 * width_multiplier),
                stride=1,
                expand_ratio=expansion_ratio
            )
        )
        # Stage 2 with MobileViT Block
        self.stage2 = nn.Sequential(
            InvertedResidual(
                make_divisible(64 * width_multiplier),
                make_divisible(128 * width_multiplier),
                stride=2,
                expand_ratio=expansion_ratio
            ),
            MobileViTBlock(
                make_divisible(128 * width_multiplier),
                make_divisible(128 * width_multiplier),
                transformer_dim=make_divisible(144 * width_multiplier),
                num_heads=num_heads,
                transformer_depth=transformer_depth,
                patch_size=patch_size,
                patch_sizes=patch_sizes,
                dropout=dropout
            )
        )

        # Stage 3 with MobileViT Block
        self.stage3 = nn.Sequential(
            InvertedResidual(
                make_divisible(128 * width_multiplier),
                make_divisible(256 * width_multiplier),
                stride=2,
                expand_ratio=expansion_ratio
            ),
            MobileViTBlock(
                make_divisible(256 * width_multiplier),
                make_divisible(256 * width_multiplier),
                transformer_dim=make_divisible(192 * width_multiplier),
                num_heads=num_heads,
                transformer_depth=transformer_depth,
                patch_size=patch_size,
                patch_sizes=patch_sizes,
                dropout=dropout
            )
        )

        # Stage 4 with MobileViT Block
        self.stage4 = nn.Sequential(
            InvertedResidual(
                make_divisible(256 * width_multiplier),
                make_divisible(512 * width_multiplier),
                stride=2,
                expand_ratio=expansion_ratio
            ),
            MobileViTBlock(
                make_divisible(512 * width_multiplier),
                make_divisible(512 * width_multiplier),
                transformer_dim=make_divisible(240 * width_multiplier),
                num_heads=num_heads,
                transformer_depth=transformer_depth,
                patch_size=patch_size,
                patch_sizes=patch_sizes,
                dropout=dropout
            )
        )

        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(make_divisible(512 * width_multiplier), num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights_stable(module):
        """Stable weight initialization"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.weight.data.abs().max() > 10:
                    m.weight.data = m.weight.data * 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Smaller initialization for linear layers
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with class weights"""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # One-hot encoding with label smoothing
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.epsilon) + self.epsilon / n_classes

        loss = (-targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Annealing with Warm Restarts"""
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch")
            if epoch >= self.T_0:
                self.T_cur = epoch % self.T_0
                self.T_i = self.T_0 * (self.T_mult ** (epoch // self.T_0))
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = np.sqrt(total_samples / (class_counts * num_classes))
    return torch.FloatTensor(weights)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Calculate comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': conf_matrix
    }

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    save_path: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_accuracy': best_acc
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['best_accuracy']


# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


# ============================================================================
# TRAINING AND VALIDATION
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    device: torch.device,
    rank: int,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    accumulation_steps: int = 4
) -> Dict:
    """Train for one epoch with NaN checking"""
    model.train()
    
    mixup = MixUp(alpha=0.2) if use_mixup else None
    cutmix = CutMix(alpha=1.0, prob=0.5) if use_cutmix else None
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    valid_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply MixUp or CutMix with reduced probability
        use_mixed = False
        if mixup is not None and random.random() < 0.3:  # Reduced from 0.5
            images, labels_a, labels_b, lam = mixup(images, labels)
            use_mixed = True
        elif cutmix is not None and random.random() < 0.3:  # Reduced from 0.5
            images, labels_a, labels_b, lam = cutmix(images, labels)
            use_mixed = True
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                logging.error(f"NaN detected in model outputs at batch {batch_idx}")
                continue
            
            if use_mixed:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                logging.error(f"NaN loss detected at batch {batch_idx}, skipping batch")
                continue
            
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            
            # Check for NaN in gradients
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logging.error(f"NaN gradient in {name}")
                        has_nan = True
                        break
            
            if has_nan:
                optimizer.zero_grad()
                logging.error("Skipping optimizer step due to NaN gradients")
                continue
            
            # Clip gradients more aggressively
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        valid_batches += 1
        
        # Store predictions for metrics (only for non-mixed batches)
        if not use_mixed:
            with torch.no_grad():
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Log progress
        if rank == 0 and batch_idx % 10 == 0:
            current_loss = loss.item() * accumulation_steps
            logging.info(
                f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] '
                f'Loss: {current_loss:.4f} '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
    
    # Calculate metrics
    avg_loss = running_loss / max(valid_batches, 1)
    
    if len(all_preds) > 0:
        train_acc = accuracy_score(all_labels, all_preds)
    else:
        train_acc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': train_acc
    }


# Fix 2: Add gradient checking function
def check_model_health(model: nn.Module) -> bool:
    """Check if model parameters and gradients are healthy"""
    for name, param in model.named_parameters():
        if param is not None:
            if torch.isnan(param).any() or torch.isinf(param).any():
                logging.error(f"NaN/Inf in parameter: {name}")
                return False
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logging.error(f"NaN/Inf in gradient: {name}")
                    return False
    return True


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        class_names
    )
    metrics['loss'] = avg_loss

    return metrics


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    save_dir: str
) -> Dict:
    """Test the model and save results"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        class_names
    )

    # Save classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )

    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Save confusion matrix
    conf_matrix_path = os.path.join(save_dir, 'confusion_matrix.npy')
    np.save(conf_matrix_path, metrics['confusion_matrix'])

    # Save predictions
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    results_path = os.path.join(save_dir, 'test_results.npz')
    np.savez(results_path, **results)

    logging.info(f"\nClassification Report:\n{report}")
    logging.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")

    return metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main_worker(rank: int, world_size: int, args):
    """Main worker function for distributed training"""

    # Setup distributed training
    setup_distributed(rank, world_size, backend='nccl')  # Use GPUs 4, 5, 6, 7
    device = torch.device(f'cuda:{rank}')

    # Setup logging (only on rank 0)
    logger = setup_logging(args.log_dir, rank)

    if rank == 0:
        logging.info("="*80)
        logging.info("Cotton Leaf Disease Detection - Optimized MobileViT")
        logging.info("="*80)
        logging.info(f"Configuration:")
        for arg, value in vars(args).items():
            logging.info(f"  {arg}: {value}")
        logging.info("="*80)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset paths
    if rank == 0:
        logging.info("Loading dataset paths...")

    all_paths, all_labels = load_dataset_paths(args.data_root, DISEASE_CLASSES)

    if rank == 0:
        logging.info(f"Total images: {len(all_paths)}")
        logging.info(f"Class distribution:")
        for idx, class_name in enumerate(DISEASE_CLASSES):
            count = all_labels.count(idx)
            logging.info(f"  {class_name}: {count}")

    # Create data splits
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_data_splits(
        all_paths, all_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    if rank == 0:
        logging.info(f"\nData splits:")
        logging.info(f"  Training: {len(train_paths)} images")
        logging.info(f"  Validation: {len(val_paths)} images")
        logging.info(f"  Test: {len(test_paths)} images")

    # Create datasets
    train_transform = get_transforms(args.image_size, is_training=True)
    val_transform = get_transforms(args.image_size, is_training=False)

    train_dataset = CottonLeafDataset(train_paths, train_labels, train_transform)
    val_dataset = CottonLeafDataset(val_paths, val_labels, val_transform)
    test_dataset = CottonLeafDataset(test_paths, test_labels, val_transform)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create model
    if rank == 0:
        logging.info("\nCreating model...")

    model = OptimizedMobileViT(
        num_classes=len(DISEASE_CLASSES),
        input_size=args.image_size,
        width_multiplier=args.width_multiplier,
        expansion_ratio=args.expansion_ratio,
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        patch_sizes=args.patch_sizes,
        dropout=args.dropout,
        stochastic_depth=args.stochastic_depth
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params / 1e6:.2f}M")
        logging.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Compute class weights
    class_weights = compute_class_weights(train_labels, len(DISEASE_CLASSES))
    class_weights = class_weights.to(device)

    # Create loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.lr_min
    )

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs
    )

    # Create gradient scaler for mixed precision
    scaler = GradScaler()

    # Load checkpoint if resuming
    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logging.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # Training loop
    if rank == 0:
        logging.info("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, epoch, device, rank,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix,
            accumulation_steps=args.accumulation_steps
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, DISEASE_CLASSES)

        # Step schedulers
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        if rank == 0:
            logging.info(f"\nEpoch [{epoch}/{args.epochs}]")
            logging.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logging.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logging.info(f"  Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            logging.info(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            logging.info(f"  Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
            logging.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_metrics['accuracy'] > best_acc
            best_acc = max(val_metrics['accuracy'], best_acc)

            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                save_checkpoint(
                    model.module, optimizer, scheduler,
                    epoch, best_acc, checkpoint_path, is_best
                )

            # Save training history
            history_path = os.path.join(args.log_dir, 'training_history.json')
            history_entry = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(history_entry)

            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

    # Final testing
    if rank == 0:
        logging.info("\n" + "="*80)
        logging.info("Training completed! Running final test evaluation...")
        logging.info("="*80)

        # Load best model for testing
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            logging.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = test(
            model.module, test_loader, device,
            DISEASE_CLASSES, args.log_dir
        )

        logging.info("\n" + "="*80)
        logging.info("FINAL TEST RESULTS")
        logging.info("="*80)
        logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        logging.info(f"Test Macro Precision: {test_metrics['macro_precision']:.4f}")
        logging.info(f"Test Macro Recall: {test_metrics['macro_recall']:.4f}")
        logging.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        logging.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        logging.info("\nPer-class metrics:")
        for idx, class_name in enumerate(DISEASE_CLASSES):
            logging.info(
                f"  {class_name}:\n"
                f"    Precision: {test_metrics['per_class_precision'][idx]:.4f}\n"
                f"    Recall: {test_metrics['per_class_recall'][idx]:.4f}\n"
                f"    F1-Score: {test_metrics['per_class_f1'][idx]:.4f}\n"
                f"    Support: {int(test_metrics['per_class_support'][idx])}"
            )
        logging.info("="*80)

        # Save final metrics to JSON
        final_metrics = {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'test_macro_precision': float(test_metrics['macro_precision']),
            'test_macro_recall': float(test_metrics['macro_recall']),
            'test_macro_f1': float(test_metrics['macro_f1']),
            'test_weighted_precision': float(test_metrics['weighted_precision']),
            'test_weighted_recall': float(test_metrics['weighted_recall']),
            'test_weighted_f1': float(test_metrics['weighted_f1']),
            'best_val_accuracy': float(best_acc),
            'per_class_metrics': {
                class_name: {
                    'precision': float(test_metrics['per_class_precision'][idx]),
                    'recall': float(test_metrics['per_class_recall'][idx]),
                    'f1_score': float(test_metrics['per_class_f1'][idx]),
                    'support': int(test_metrics['per_class_support'][idx])
                }
                for idx, class_name in enumerate(DISEASE_CLASSES)
            }
        }

        metrics_path = os.path.join(args.log_dir, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

        logging.info(f"\nFinal metrics saved to {metrics_path}")
        logging.info("Training and evaluation pipeline completed successfully!")

    # Cleanup
    cleanup_distributed()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Optimized MobileViT for Cotton Leaf Disease Detection'
    )

    # Data parameters
    parser.add_argument('--data_root', type=str, default='ds',
                        help='Root directory with "aug_ds" and "orig_ds" subfolders')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for model checkpoints')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (default: 224)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per GPU (default: 128)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate (default: 0.05)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')

    # Scheduler parameters
    parser.add_argument('--T_0', type=int, default=30,
                        help='Cosine annealing T_0 (default: 30)')
    parser.add_argument('--T_mult', type=int, default=2,
                        help='Cosine annealing T_mult (default: 2)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')

    # Data split parameters
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation data ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test data ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Model architecture parameters
    parser.add_argument('--width_multiplier', type=float, default=1.2,
                        help='Width multiplier for model (default: 1.2)')
    parser.add_argument('--expansion_ratio', type=int, default=4,
                        help='Expansion ratio for inverted residual (default: 4)')
    parser.add_argument('--transformer_depth', type=int, default=2,
                        help='Transformer depth (default: 2)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--patch_size', type=int, default=2,
                        help='Base patch size (default: 2)')
    parser.add_argument('--patch_sizes', type=int, nargs='+', default=[2, 4, 8],
                        help='Multi-scale patch sizes (default: [2, 4, 8])')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--stochastic_depth', type=float, default=0.1,
                        help='Stochastic depth rate (default: 0.1)')

    # Loss and augmentation parameters
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of Label Smoothing')
    parser.add_argument('--use_mixup', action='store_true', default=True,
                        help='Use MixUp augmentation during training')
    parser.add_argument('--use_cutmix', action='store_true', default=True,
                        help='Use CutMix augmentation during training')

    # Training optimization parameters
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')

    # Checkpointing parameters
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    # Distributed training parameters
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set world size (4 GPUs: 4, 5, 6, 7)
    world_size = 4

    # Set CUDA device visibility to use GPUs 4, 5, 6, 7
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    # Launch distributed training
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        main_worker(0, 1, args)
