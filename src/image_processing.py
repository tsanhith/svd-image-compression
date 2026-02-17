import numpy as np
from PIL import Image

def get_image_channels(image_pil, is_grayscale):
    """
    Converts a PIL image to a list of numpy arrays (channels).

    Args:
        image_pil (PIL.Image.Image): The input image.
        is_grayscale (bool): Flag to convert to grayscale.

    Returns:
        tuple: A tuple containing the list of channels and the processed PIL image.
    """
    if is_grayscale:
        processed_pil = image_pil.convert('L')
        image_np = np.array(processed_pil, dtype=np.float32)
        channels = [image_np]
    else:
        processed_pil = image_pil.convert('RGB')
        image_np = np.array(processed_pil, dtype=np.float32)
        channels = [image_np[:, :, i] for i in range(3)]
    
    return channels, processed_pil

def perform_svd(channels):
    """
    Performs SVD on a list of image channels.

    Args:
        channels (list): A list of numpy arrays, each representing an image channel.

    Returns:
        tuple: Three lists containing U, s, and Vh matrices for each channel.
    """
    U_all, s_all, Vh_all = [], [], []
    for channel in channels:
        U, s, Vh = np.linalg.svd(channel, full_matrices=False)
        U_all.append(U)
        s_all.append(s)
        Vh_all.append(Vh)
    return U_all, s_all, Vh_all

def reconstruct_images(U_all, s_all, Vh_all, k, is_grayscale):
    """
    Reconstructs images from SVD components for both most and least important ranks.

    Args:
        U_all, s_all, Vh_all (list): SVD components for each channel.
        k (int): The number of singular values to use.
        is_grayscale (bool): True if the image is grayscale.

    Returns:
        tuple: A tuple containing the reconstructed image and the least important ranks image as PIL Images.
    """
    reconstructed_channels = []
    least_important_channels = []
    max_k = len(s_all[0])

    for i in range(len(U_all)):
        U, s, Vh = U_all[i], s_all[i], Vh_all[i]
        
        # Most important ranks
        reconstructed_matrix = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
        reconstructed_channels.append(reconstructed_matrix)
        
        # Least important ranks
        if k < max_k:
            least_matrix = U[:, -k:] @ np.diag(s[-k:]) @ Vh[-k:, :]
        else:
            least_matrix = np.zeros_like(U[:,0:1]) # Fallback for k=max_k
        least_important_channels.append(least_matrix)

    # Combine channels and convert to PIL Image
    def to_pil(channels_list):
        if is_grayscale:
            img_np = channels_list[0]
        else:
            img_np = np.stack(channels_list, axis=-1)
        
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    reconstructed_image_pil = to_pil(reconstructed_channels)
    least_important_image_pil = to_pil(least_important_channels)
    
    return reconstructed_image_pil, least_important_image_pil

def calculate_metrics(height, width, num_channels, U_all, s_all, Vh_all, k):
    """
    Calculates energy preserved, storage savings, and compression ratio.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Assuming float32 which is 4 bytes per value
    bytes_per_value = 4

    # Storage and Compression
    original_pixels = height * width * num_channels
    original_storage_bytes = original_pixels * bytes_per_value

    compressed_data_values = sum(U.shape[0] * k + k + Vh.shape[1] * k for U, Vh in zip(U_all, Vh_all))
    compressed_storage_bytes = compressed_data_values * bytes_per_value
    
    storage_savings = 100 * (1 - (compressed_storage_bytes / original_storage_bytes)) if original_storage_bytes > 0 else 0
    compression_factor = original_storage_bytes / compressed_storage_bytes if compressed_storage_bytes > 0 else 0

    return {
        "original_storage_bytes": original_storage_bytes,
        "compressed_storage_bytes": compressed_storage_bytes,
        "storage_savings": storage_savings,
        "compression_factor": compression_factor
    }


def calculate_psnr(original_image_np, reconstructed_image_np):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        original_image_np (np.ndarray): Original image array in [0, 255].
        reconstructed_image_np (np.ndarray): Reconstructed image array in [0, 255].

    Returns:
        float: PSNR value in decibels. Returns inf for perfect reconstruction.
    """
    original = original_image_np.astype(np.float32)
    reconstructed = reconstructed_image_np.astype(np.float32)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def find_k_for_energy(s_all, target_energy_percent):
    """
    Finds the smallest k that reaches a target cumulative energy across channels.

    Args:
        s_all (list[np.ndarray]): Singular values for each channel.
        target_energy_percent (float): Target energy percentage in [0, 100].

    Returns:
        int: Smallest k meeting the target for all channels.
    """
    target = np.clip(target_energy_percent, 0, 100) / 100.0
    max_k = len(s_all[0])
    channel_ks = []

    for s in s_all:
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        idx = int(np.searchsorted(energy, target, side="left"))
        channel_ks.append(min(idx + 1, max_k))

    return max(channel_ks)


def summarize_quality_levels(U_all, s_all, Vh_all, is_grayscale, original_image_np, k_levels):
    """
    Builds a quality/compression summary for selected rank levels.

    Args:
        U_all, s_all, Vh_all (list): SVD components.
        is_grayscale (bool): Grayscale flag.
        original_image_np (np.ndarray): Original image as numpy array.
        k_levels (list[int]): Rank values to evaluate.

    Returns:
        list[dict]: Rows with k, psnr, compression and storage metrics.
    """
    height, width = original_image_np.shape[:2]
    num_channels = 1 if is_grayscale else 3
    rows = []

    for k in sorted(set(k_levels)):
        recon_pil, _ = reconstruct_images(U_all, s_all, Vh_all, k, is_grayscale)
        recon_np = np.array(recon_pil, dtype=np.float32)
        metrics = calculate_metrics(height, width, num_channels, U_all, s_all, Vh_all, k)
        psnr = calculate_psnr(original_image_np, recon_np)
        rows.append({
            "k": k,
            "psnr_db": psnr,
            "compression_factor": metrics["compression_factor"],
            "storage_savings": metrics["storage_savings"],
            "compressed_storage_bytes": metrics["compressed_storage_bytes"],
        })

    return rows
