"""
Parts of an Attention U-Net with a Residual Encoder and ASPP Bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import Tuple, List

# --- 1. NEW: Residual Block (Replaces DoubleConv) ---
# Implements Improvement 1: Residual connections

class ResidualBlock(nn.Module):
    """(conv => [BN] => ReLU => conv => [BN]) + shortcut => ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # 1x1 conv to project input to the same channel dim as output
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut *before* the final activation
        out += identity
        out = self.relu(out)
        
        return out


# --- 2. NEW: ASPP Module (For Bottleneck) ---
# Implements Improvement 3: Multi-Scale Context (ASPP)

class _ASPPConv(nn.Module):
    """Helper for a single ASPP branch."""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.module(x)

class _ASPPPool(nn.Module):
    """Helper for the Global Average Pooling branch."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.module(x)
        # Upsample back to the original feature map size
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    For a 512x512 input, the bottleneck (H/32) is 16x16.
    Rates (3, 6, 9) are chosen to capture regional and near-global context
    on this 16x16 feature map.
    """
    def __init__(self, in_channels, out_channels, rates: Tuple[int, ...] = (3, 6, 9)):
        super().__init__()
        
        # Intermediate channels for each branch
        inter_channels = in_channels // 4 
        
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.b_rates = nn.ModuleList([
            _ASPPConv(in_channels, inter_channels, rate) for rate in rates
        ])
        self.b_pool = _ASPPPool(in_channels, inter_channels)
        
        # Total channels after concatenation: 1 (1x1) + len(rates) + 1 (pool)
        total_channels = inter_channels * (2 + len(rates))
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branches = [self.b1(x)]
        branches.extend([b(x) for b in self.b_rates])
        branches.append(self.b_pool(x))
        
        x = torch.cat(branches, dim=1)
        x = self.conv_out(x)
        return x


# --- 3. NEW: Attention Gate (For Skip Connections) ---
# Implements Improvement 2: Spatial Attention

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for U-Net skip connections.
    Filters the features from the encoder (x) using context
    from the decoder (g).
    """
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from decoder (lower res, upsampled)
        # x: skip connection from encoder (higher res)
        g_out = self.W_g(g)
        x_out = self.W_x(x)
        
        # Add and apply activation
        psi_in = self.relu(g_out + x_out)
        
        # Compute attention coefficients (alpha)
        alpha = self.psi(psi_in) # [B, 1, H, W]
        
        # Apply attention to the encoder features
        # Broadcasting alpha across the channel dimension of x
        return x * alpha


# --- 4. MODIFIED: Core U-Net Parts ---

class Down(nn.Module):
    """Downscaling with maxpool then conv block"""
    def __init__(self, in_channels, out_channels, conv_block=ResidualBlock):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv, now with optional Attention Gate"""
    def __init__(self, in_channels, out_channels, bilinear=True, 
                 conv_block=ResidualBlock, attention_gate: AttentionGate = None):
        super().__init__()
        self.attention_gate = attention_gate

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # `in_channels` is the total concatenated channels
            self.conv = conv_block(in_channels, out_channels, in_channels // 2)
        else:
            # `in_channels` here is from the upsampled path *before* cat
            # This logic is based on your original `Up` class.
            # `in_channels` // 2 matches the `factor` logic.
            up_in_ch = in_channels // 2 if attention_gate else in_channels
            self.up = nn.ConvTranspose2d(up_in_ch, up_in_ch // 2, kernel_size=2, stride=2)
            self.conv = conv_block(in_channels, out_channels)
            
        # This fixes a potential bug from your original code if not bilinear
        self.bilinear = bilinear

    def forward(self, x1, x2):
        # x1: from decoder path (e.g., [B, 512, 16, 16])
        # x2: from encoder skip connection (e.g., [B, 512, 32, 32])
        
        x1_up = self.up(x1)
        
        # Apply Attention Gate if it exists
        if self.attention_gate:
            # x1_up is the gating signal 'g'
            # x2 is the skip signal 'x'
            x2 = self.attention_gate(g=x1_up, x=x2)
        
        # Pad x1_up to match x2's size
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x)


# --- 5. REBUILT: Main U-Net Model ---

class RobustUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, chs: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)):
        super(RobustUNet, self).__init__()
        
        self.n_channels = in_ch
        self.n_classes = out_ch
        bilinear = True # Sticking to your original's hardcoded value
        
        # conv_block to use everywhere
        conv_block = ResidualBlock
        
        # Channel factor based on bilinear
        # Note: Your original `chs` tuple and `factor` logic was a bit confusing.
        # This implementation follows your original channel math *exactly*.
        # `chs[5]` (1024) is the concatenated channel count for up1
        # `chs[4]` (512) is the concatenated channel count for up2
        # `chs[i] // factor` is the output channel count of the decoder block
        
        factor = 2 if bilinear else 1

        # --- Encoder (Residual) ---
        self.inc = conv_block(self.n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1], conv_block=conv_block) # H/2
        self.down2 = Down(chs[1], chs[2], conv_block=conv_block) # H/4
        self.down3 = Down(chs[2], chs[3], conv_block=conv_block) # H/8
        self.down4 = Down(chs[3], chs[4], conv_block=conv_block) # H/16

        # --- Bottleneck (ASPP) ---
        # Input to bottleneck is chs[4] (512). Output is chs[5] // factor (512).
        # ASPP Rates (3, 6, 9) are for the 16x16 (H/32) feature map.
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ASPPModule(in_channels=chs[4], out_channels=chs[5] // factor, rates=(3, 6, 9))
        ) # H/32

        # --- Decoder (Attention) ---
        
        # Attention Gate for up1
        # g comes from bottleneck (512 ch), x comes from down4 (512 ch)
        self.att1 = AttentionGate(g_channels=chs[5] // factor, 
                                  x_channels=chs[4], 
                                  inter_channels=chs[4] // 2)
        # Up(concatenated_channels, output_channels, ...)
        self.up1 = Up(in_channels=chs[5], out_channels=chs[4] // factor, 
                      bilinear=bilinear, conv_block=conv_block, attention_gate=self.att1)
        
        # Attention Gate for up2
        # g comes from up1 (256 ch), x comes from down3 (256 ch)
        self.att2 = AttentionGate(g_channels=chs[4] // factor, 
                                  x_channels=chs[3], 
                                  inter_channels=chs[3] // 2)
        # Up(concatenated_channels, output_channels, ...)
        self.up2 = Up(in_channels=chs[4], out_channels=chs[3] // factor, 
                      bilinear=bilinear, conv_block=conv_block, attention_gate=self.att2)
        
        # Final output convolution
        self.outc = OutConv(chs[3] // factor, out_ch)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bottleneck
        x6 = self.bottleneck(x5)
        
        # Decoder
        # up1 takes x6 (from bottleneck) and x5 (skip)
        x = self.up1(x6, x5)
        # up2 takes x (from up1) and x4 (skip)
        x = self.up2(x, x4)
        
        # Output
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = RobustUNet(in_ch=1, out_ch=90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("--- Model Summary (RobustUNet with ResBlocks, ASPP, and Attention) ---")
    # Using 512x512 as requested
    summary(model, (1, 512, 512))