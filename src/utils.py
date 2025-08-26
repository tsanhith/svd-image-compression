import io
from PIL import Image

def resize_image(image, max_size=1024):
    """
    Resizes a PIL image to a maximum side length while preserving aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        max_size (int): The maximum width or height.

    Returns:
        PIL.Image.Image: The resized image.
    """
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def image_to_bytes(image):
    """
    Converts a PIL image to a byte stream for downloading.

    Args:
        image (PIL.Image.Image): The image to convert.

    Returns:
        bytes: The image data as bytes.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def format_bytes(size_bytes):
    """
    Formats a size in bytes to a human-readable string (KB, MB, GB).

    Args:
        size_bytes (int): The size in bytes.

    Returns:
        str: The formatted size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"
