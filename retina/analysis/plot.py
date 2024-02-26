import matplotlib.pyplot as plt


def plot_spatial_rfs(model_rfs, rows, cols, range_all=True):
    # model_rfs: batch x rf_len x w x h
    #assert model_rfs.shape[0] == rows * cols

    model_spatial_rfs = model_rfs
    ranges = model_spatial_rfs.abs().amax(dim=(1, 2))

    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    n = rows * cols
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols

        vmin = -ranges.abs().max() if range_all else -ranges[i]
        vmax = ranges.abs().max() if range_all else ranges[i]
        if vmin == vmax:  # Ensure to plot a blank square if RF is zero
            vmin, vmax = -1, 1
        axs[row_idx, col_idx].imshow(model_spatial_rfs[i], cmap='bwr', vmin=vmin, vmax=vmax)
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