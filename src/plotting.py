import matplotlib.pyplot as plt
import numpy as np

def plot_singular_values(s_all_channels, is_grayscale):
    """
    Plots the singular values for each color channel.

    Args:
        s_all_channels (list): A list of singular value arrays for each channel.
        is_grayscale (bool): True if the image is grayscale.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_grayscale:
        ax.plot(s_all_channels[0], label='Grayscale Channel', color='black')
    else:
        colors = ['red', 'green', 'blue']
        labels = ['Red Channel', 'Green Channel', 'Blue Channel']
        for i in range(3):
            ax.plot(s_all_channels[i], color=colors[i], label=labels[i])
    
    ax.set_title("Singular Values (Σ)", fontsize=16)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Singular Value Magnitude", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    return fig

def plot_cumulative_energy(s_all_channels, is_grayscale):
    """
    Plots the cumulative energy captured by the singular values.

    Args:
        s_all_channels (list): A list of singular value arrays for each channel.
        is_grayscale (bool): True if the image is grayscale.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def plot_channel_energy(s, color, label):
        s_squared = s**2
        cumulative_energy = np.cumsum(s_squared) / np.sum(s_squared) * 100
        ax.plot(cumulative_energy, color=color, label=label)

    if is_grayscale:
        plot_channel_energy(s_all_channels[0], 'black', 'Grayscale Channel')
    else:
        colors = ['red', 'green', 'blue']
        labels = ['Red Channel', 'Green Channel', 'Blue Channel']
        for i in range(3):
            plot_channel_energy(s_all_channels[i], colors[i], labels[i])

    ax.set_title("Cumulative Energy Captured by Singular Values", fontsize=16)
    ax.set_xlabel("Number of Singular Values (k)", fontsize=12)
    ax.set_ylabel("Cumulative Energy (%)", fontsize=12)
    ax.grid(True, ls="--")
    ax.legend()
    return fig

def plot_difference_heatmap(original_image_np, reconstructed_image_np, is_grayscale):
    """
    Computes and plots the difference between two images as a heatmap.

    Args:
        original_image_np (np.ndarray): The original image array.
        reconstructed_image_np (np.ndarray): The reconstructed image array.
        is_grayscale (bool): True if the image is grayscale.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    diff_image_np = np.abs(original_image_np.astype(np.float32) - reconstructed_image_np.astype(np.float32))
    if not is_grayscale:
        diff_image_np = diff_image_np.mean(axis=2) # Average channels for a single heatmap
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(diff_image_np, cmap='hot')
    fig.colorbar(heatmap, ax=ax)
    ax.set_title("Information Loss (Heatmap)")
    ax.axis('off')
    return fig
