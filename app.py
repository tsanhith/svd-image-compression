import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

# Function to compress image using SVD
def compress_image_svd(image, k):
    # Convert image to grayscale if it's colored
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    
    # Apply SVD
    U, S, VT = np.linalg.svd(img_gray, full_matrices=False)
    
    # Reconstruct with top k singular values
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    compressed_img = np.dot(U_k, np.dot(S_k, VT_k))
    
    return compressed_img

# Function to reconstruct least important k singular values
def reconstruct_least_important(image, k):
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    
    U, S, VT = np.linalg.svd(img_gray, full_matrices=False)
    
    # Reconstruct image using the *last* k singular values
    S_least = np.zeros((len(S), len(S)))
    np.fill_diagonal(S_least, S)
    S_least[:-k, :-k] = 0  # Keep only the last k singular values
    
    least_important_img = np.dot(U, np.dot(S_least, VT))
    
    return least_important_img

# Function to convert image to bytes for download
def image_to_bytes(img):
    img = Image.fromarray(np.uint8(img))
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Streamlit App UI
st.title("📸 Image Compression using Singular Value Decomposition (SVD)")
st.write("Upload an image and see how SVD can compress and reconstruct it.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # Read color image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(img_rgb, use_container_width=True)

    # Slider for number of singular values
    k = st.slider("Select number of singular values (k)", min_value=1, max_value=min(img.shape[0], img.shape[1]), value=50)

    # Compress using SVD
    compressed_img = compress_image_svd(img, k)

    # Display compressed image
    st.subheader("Compressed Image")
    st.image(compressed_img, use_container_width=True, caption=f"Reconstructed with the top {k} singular values")

    # Reconstruct using least important k singular values
    least_important_img = reconstruct_least_important(img, k)

    # Show side by side
    st.subheader("Comparison")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.image(compressed_img, caption=f"Image from First {k} Ranks (Most Important)", use_container_width=True)
    with comp_col2:
        st.image(least_important_img, caption=f"Image from Last {k} Ranks (Least Important)", use_container_width=True)

    # Download buttons
    st.subheader("Download Images")
    st.download_button(
       label="Download Compressed Image",
       data=image_to_bytes(compressed_img),
       file_name=f"compressed_k{k}.png",
       mime="image/png"
    )
    st.download_button(
       label="Download 'Least Important' Image",
       data=image_to_bytes(least_important_img),
       file_name=f"least_important_k{k}.png",
       mime="image/png"
    )
