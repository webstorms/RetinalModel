import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()

    kernel = torch.outer(gauss, gauss)
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(input_tensor, kernel_size=5, sigma=1.0):
    channels = input_tensor.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(input_tensor.device)
    padding = kernel_size // 2
    return F.conv2d(input_tensor, kernel, padding=padding, groups=channels)


def plot_spatial_rfs(model_rfs, rows, cols, range_all=True, sigma=0):
    # model_rfs: batch x 1 x w x h or (batch, h, w) in NumPy or PyTorch format
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    # If model_rfs is a NumPy array, convert to tensor for Gaussian convolution
    if isinstance(model_rfs, np.ndarray):
        model_rfs = torch.from_numpy(model_rfs).float()

    # Ensure correct shape (batch x 1 x W x H) for convolution
    if model_rfs.ndimension() == 3:  # (batch, H, W)
        model_rfs = model_rfs.unsqueeze(1)  # Add channel dimension: (batch, 1, H, W)

    model_rfs = model_rfs.detach().cpu()

    # Debugging line to check if sigma=0 is handled correctly
    print(f"Sigma value: {sigma}")

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        print(f"Applying Gaussian blur with sigma={sigma}")
        model_rfs = gaussian_blur(model_rfs, kernel_size=5, sigma=sigma)
    else:
        print("No Gaussian blur applied (sigma=0).")

    # Convert back to NumPy for plotting
    model_rfs = model_rfs.squeeze(1).numpy()  # Shape: (batch, H, W)

    # Calculate the max absolute value per receptive field (use np.amax for NumPy arrays)
    # Ensure correct handling of model_rfs.shape (batch, height, width)
    if model_rfs.ndim == 3:  # (batch, H, W)
        ranges = np.amax(np.abs(model_rfs), axis=(1, 2))  # Find max range per RF
    else:
        raise ValueError(f"Expected model_rfs to have 3 dimensions (batch, H, W), but got {model_rfs.ndim}.")

    n = rows * cols
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols

        vmin = -np.max(ranges) if range_all else -ranges[i]
        vmax = np.max(ranges) if range_all else ranges[i]
        if vmin == vmax:  # Ensure to plot a blank square if RF is zero
            vmin, vmax = -1, 1

        axs[row_idx, col_idx].imshow(model_rfs[i], cmap='bwr', vmin=vmin, vmax=vmax)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx].text(0, 0, f"{i}", color='black', fontsize=10, alpha=0.7)

    fig.tight_layout()
    plt.show()

def plot_average_rfs(model_rfs, rows, cols, range_all=True, sigma=0):
    # model_rfs: batch x 1 x w x h or (batch, h, w) in NumPy or PyTorch format
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    # If model_rfs is a NumPy array, convert to tensor for Gaussian convolution
    if isinstance(model_rfs, np.ndarray):
        model_rfs = torch.from_numpy(model_rfs).float()

    # Ensure correct shape (batch x 1 x W x H) for convolution
    if model_rfs.ndimension() == 3:  # (batch, H, W)
        model_rfs = model_rfs.unsqueeze(1)  # Add channel dimension: (batch, 1, H, W)

    model_rfs = model_rfs.detach().cpu()

    # Debugging line to check if sigma=0 is handled correctly
    print(f"Sigma value: {sigma}")

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        print(f"Applying Gaussian blur with sigma={sigma}")
        model_rfs = gaussian_blur(model_rfs, kernel_size=5, sigma=sigma)
    else:
        print("No Gaussian blur applied (sigma=0).")

    # Convert back to NumPy for plotting
    model_rfs = model_rfs.squeeze(1).numpy()  # Shape: (batch, H, W)

    # Calculate the max absolute value per receptive field (use np.amax for NumPy arrays)
    # Ensure correct handling of model_rfs.shape (batch, height, width)
    if model_rfs.ndim == 3:  # (batch, H, W)
        ranges = np.amax(np.abs(model_rfs), axis=(1, 2))  # Find max range per RF
    else:
        raise ValueError(f"Expected model_rfs to have 3 dimensions (batch, H, W), but got {model_rfs.ndim}.")

    n = rows * cols
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols

        vmin = -np.max(ranges) if range_all else -ranges[i]
        vmax = np.max(ranges) if range_all else ranges[i]
        if vmin == vmax:  # Ensure to plot a blank square if RF is zero
            vmin, vmax = -1, 1

        axs[row_idx, col_idx].imshow(model_rfs[i], cmap='seismic', vmin=vmin, vmax=vmax)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx].text(0, 0, f"{i}", color='black', fontsize=10, alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_spatial_rfs_smooth(model_rfs, rows, cols, range_all=True, sigma=1):
    # model_rfs: batch x rf_len x w x h
    #assert model_rfs.shape[0] == rows * cols

    model_spatial_rfs = model_rfs
    ranges = model_spatial_rfs.abs().amax(dim=(1, 2))

    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    n = rows * cols
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        smooth_spatial_rf = gaussian_filter(model_spatial_rfs[i], sigma=sigma)

        vmin = -ranges.abs().max() if range_all else -ranges[i]
        vmax = ranges.abs().max() if range_all else ranges[i]
        if vmin == vmax:  # Ensure to plot a blank square if RF is zero
            vmin, vmax = -1, 1
        axs[row_idx, col_idx].imshow(smooth_spatial_rf, cmap='seismic', vmin=vmin, vmax=vmax)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx].text(0,0,f"{i}")
    fig.tight_layout()

def plot_spatiotemporal_rf(spatiotemporal_rf, range_val=None):
    # spatiotemporal_rf: rf_len x w x h

    spatiotemporal_rf = spatiotemporal_rf.cpu().detach()
    if range_val is None:
        range_val = spatiotemporal_rf.abs().max()
    fig, axs = plt.subplots(1, len(spatiotemporal_rf), figsize=(10, 10))

    for i in range(len(spatiotemporal_rf)):
        im = axs[i].imshow(spatiotemporal_rf[i], cmap='bwr', vmin=-range_val, vmax=range_val)
        axs[i].axis('off')

    cbar_ax = fig.add_axes([1.02, 0.4, 0.02, 0.2])
    bar = plt.colorbar(im, cax=cbar_ax)
    bar.ax.get_yaxis().labelpad = 15
    bar.ax.tick_params(labelsize='12')