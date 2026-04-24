"""
PancrAI — TransUNet Architecture
Transformer + U-Net hybrid for medical image segmentation.

Architecture:
  - Encoder: ResNet50 pretrained CNN feature extractor
  - Bottleneck: Vision Transformer (ViT) with 12 layers
  - Decoder: U-Net-style decoder with attention gates
  - Output: Binary segmentation mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from einops import rearrange


# ─── Vision Transformer Components ──────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Divide feature map into patches and embed them."""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int,
                 img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)                             # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')    # (B, N, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)        # (B, N+1, embed_dim)
        x = x + self.pos_embed
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with dropout."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer bottleneck (12 layers, 768-dim, 12 heads)."""

    def __init__(self, in_channels: int = 1024, img_size: int = 14,
                 patch_size: int = 1, embed_dim: int = 768,
                 depth: int = 12, n_heads: int = 12, mlp_dim: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels, patch_size, embed_dim, img_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.n_patches = img_size * img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sequence of patch tokens (excluding cls token)."""
        x = self.patch_embed(x)                      # (B, N+1, embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 1:, :]                              # remove cls token
        B, N, C = x.shape
        h = w = int(math.sqrt(N))
        x = x.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# ─── Attention Gate ──────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention gate for U-Net skip connections.
    Filters relevant features from the encoder path.
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear',
                               align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single decoder block: upsample, attention gate, concatenate, conv×2.
    """

    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.attn = AttentionGate(out_channels, skip_channels, out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                               align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── Main TransUNet Model ─────────────────────────────────────────────────────

class TransUNet(nn.Module):
    """
    TransUNet: Transformer-in-U-Net for Medical Image Segmentation.

    Architecture:
      CNN Encoder (ResNet50) → ViT Bottleneck → U-Net Decoder
      with attention-gated skip connections.

    Args:
        img_size: Input image size (224 or 512). Default 224.
        in_channels: Number of input channels. Default 3 (RGB).
        num_classes: Segmentation classes. Default 1 (binary).
        pretrained: Load ResNet50 pretrained on ImageNet. Default True.
    """

    def __init__(self, img_size: int = 224, in_channels: int = 3,
                 num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.img_size = img_size

        # ── CNN Encoder (ResNet50 backbone) ──
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(weights=None)

        # Adjust first conv for grayscale if needed
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)

        # Extract encoder stages
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu)          # → 64ch, H/2
        self.pool0 = backbone.maxpool                     # → 64ch, H/4
        self.enc1 = backbone.layer1                       # → 256ch, H/4
        self.enc2 = backbone.layer2                       # → 512ch, H/8
        self.enc3 = backbone.layer3                       # → 1024ch, H/16

        # ── Vision Transformer bottleneck ──
        vit_img_size = img_size // 16                     # 14 for 224 input
        self.vit = VisionTransformer(
            in_channels=1024,
            img_size=vit_img_size,
            patch_size=1,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_dim=3072,
            dropout=0.1,
        )
        self.vit_proj = nn.Conv2d(768, 512, kernel_size=1)  # project ViT → decoder

        # ── U-Net Decoder with attention gates ──
        # Each DecoderBlock(in, skip, out)
        self.dec3 = DecoderBlock(512, 1024, 256)   # ViT output + enc3 skip
        self.dec2 = DecoderBlock(256, 512,  128)   # + enc2 skip
        self.dec1 = DecoderBlock(128, 256,  64)    # + enc1 skip
        self.dec0 = DecoderBlock(64,  64,   32)    # + enc0 skip

        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # ── Encode ──
        e0 = self.enc0(x)           # (B, 64,   H/2,  W/2)
        p0 = self.pool0(e0)         # (B, 64,   H/4,  W/4)
        e1 = self.enc1(p0)          # (B, 256,  H/4,  W/4)
        e2 = self.enc2(e1)          # (B, 512,  H/8,  W/8)
        e3 = self.enc3(e2)          # (B, 1024, H/16, W/16)

        # ── ViT bottleneck ──
        vit_out = self.vit(e3)      # (B, 768, H/16, W/16)
        vit_out = self.vit_proj(vit_out)  # (B, 512, H/16, W/16)

        # ── Decode ──
        d3 = self.dec3(vit_out, e3)  # (B, 256, H/8,  W/8)
        d2 = self.dec2(d3, e2)       # (B, 128, H/4,  W/4)
        d1 = self.dec1(d2, e1)       # (B, 64,  H/4,  W/4)
        d0 = self.dec0(d1, e0)       # (B, 32,  H/2,  W/2)

        # ── Final output ──
        out = self.final_up(d0)      # (B, 32,  H, W)
        out = self.seg_head(out)     # (B, 1,   H, W)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated prediction mask."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_transunet(img_size: int = 224, pretrained: bool = True,
                    weights_path: str = None) -> TransUNet:
    """
    Build and optionally load a TransUNet model.

    Args:
        img_size: Input resolution.
        pretrained: Load ImageNet pretrained ResNet50 encoder.
        weights_path: Path to fine-tuned checkpoint (.pth). If None or not
                      found, returns model with random / ImageNet weights.

    Returns:
        TransUNet model in eval mode.
    """
    import os
    model = TransUNet(img_size=img_size, pretrained=pretrained)

    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location="cpu")
            # Handle DataParallel / Lightning checkpoints
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            print(f"[TransUNet] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[TransUNet] Warning — could not load weights: {e}")
            print("[TransUNet] Running in demo mode with random weights.")
    else:
        print("[TransUNet] No weights file found — running in demo mode.")

    model.eval()
    return model
