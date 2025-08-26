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
