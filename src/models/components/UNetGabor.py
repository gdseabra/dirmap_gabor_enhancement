""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.components.UNet import UNet

import torch
import torch.nn as nn
import numpy as np
import cv2

def generate_gabor_kernels(num_orientations, ksize, sigma, lambd, gamma=0.5):
    """
    Generates a stack of Gabor filter kernels.

    Args:
        num_orientations (int): Number of orientations (e.g., 90).
        ksize (int): The size of the Gabor kernel (e.g., 31).
        sigma (float): Standard deviation of the Gaussian envelope.
        lambd (float): Wavelength of the sinusoidal factor.
        gamma (float): Spatial aspect ratio.

    Returns:
        torch.Tensor: A tensor of shape (num_orientations, 1, ksize, ksize)
                      containing the Gabor kernels.
    """
    kernels = []
    # Orientations from 0 to 178 degrees, matching your U-Net output
    for i in range(num_orientations):
        theta = ((90 + 2*i)/180) * np.pi  # Angle in radians
        theta = np.arctan(-np.sin(theta)/np.cos(theta))
        kernel = cv2.getGaborKernel(
            (ksize, ksize), 
            sigma, 
            theta, 
            lambd, 
            gamma, 
            psi=0, # Phase offset, 0 and pi/2 are common
            ktype=cv2.CV_32F
        )
        # Add a channel dimension for PyTorch compatibility
        kernels.append(kernel)
    
    # Stack kernels into a single tensor
    gabor_kernels = np.stack(kernels, axis=0)
    # Add the 'in_channels' dimension
    gabor_kernels = torch.from_numpy(gabor_kernels).unsqueeze(1)
    
    return gabor_kernels

class GaborConvLayer(nn.Module):
    def __init__(self, num_orientations=90, ksize=31, sigma=4.0, lambd=10.0):
        super(GaborConvLayer, self).__init__()
        
        # Generate the fixed Gabor kernels
        gabor_weights = generate_gabor_kernels(num_orientations, ksize, sigma, lambd)
        
        # Create a non-trainable Conv2d layer
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=num_orientations, 
            kernel_size=ksize, 
            padding='same', # Preserves input spatial dimensions
            bias=False
        )
        
        # Assign the fixed Gabor weights and make them non-trainable
        self.conv.weight = nn.Parameter(gabor_weights, requires_grad=False)

    def forward(self, x):
        # Apply the convolution
        return self.conv(x)

class UNetGabor(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        super(UNetGabor, self).__init__()
        
        self.dirmap_net = UNet(in_ch=1, out_ch=90, ndim=ndim, chs=chs)

        self.gabor_layer = GaborConvLayer(
            num_orientations=90, 
            ksize=25, 
            sigma=4.5, 
            lambd=8.0
        )

    def forward(self, x):
    
        # --- Step 1: Get orientation map from U-Net ---
        # out_dirmap shape: (B, C, H, W), where C is the number of orientations
        out_dirmap = self.dirmap_net(x)

        # --- Create a hard mask from the orientation map ---
        # Find the index of the maximum value for each pixel along the channel dimension.
        # max_indices shape: (B, H, W)
        max_indices = torch.argmax(out_dirmap, dim=1)
        
        # Create a one-hot encoded tensor from the indices. This produces a tensor
        # of shape (B, H, W, C), where the last dimension is the one-hot vector.
        hard_mask = F.one_hot(max_indices, num_classes=out_dirmap.shape[1])
        
        # Permute dimensions to (B, C, H, W) to match the gabor_responses shape
        # and cast to float for multiplication.
        hard_mask = hard_mask.permute(0, 3, 1, 2).float()

        hard_mask_upscaled = F.interpolate(hard_mask, scale_factor=8, mode="nearest")

        # --- Step 2: Get Gabor filter responses ---
        # gabor_responses shape: (B, C, H, W)
        gabor_responses = self.gabor_layer(x)
        
        # --- Step 3: Weight Gabor responses by the hard mask ---
        # Element-wise multiplication. For each pixel, only the Gabor response
        # corresponding to the max orientation score will be kept. All others will be zero.
        weighted_responses = hard_mask_upscaled * gabor_responses
        
        # --- Step 4: Sum the weighted responses ---
        # Summing across the channel dimension effectively selects the single non-zero
        # Gabor response for each pixel, creating the final enhanced feature map.
        # out_enh shape: (B, 1, H, W)
        out_enh = torch.sum(weighted_responses, dim=1, keepdim=True)
        
        return out_dirmap, out_enh




if __name__ == '__main__':
    model         =  UNetGabor()

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    summary(model, (1, 256, 256))