import streamlit as st
from PIL import Image
import numpy as np
import time

# Import functions from your src folder
from src.utils import resize_image, image_to_bytes, format_bytes
from src.plotting import plot_singular_values, plot_cumulative_energy, plot_difference_heatmap
from src.image_processing import get_image_channels, perform_svd, reconstruct_images, calculate_metrics
from src.ui_components import display_svd_explanation

# --- Page Configuration ---
st.set_page_config(
    page_title="SVD Image Explorer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖼️ SVD Image Explorer")
st.write("""
Welcome! This app helps you interactively explore **Singular Value Decomposition (SVD)** for image compression. 
Upload an image, adjust the number of singular values (`k`), and see how the image is reconstructed.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        is_grayscale = st.checkbox("Convert to Grayscale")

        # Load and process the image
        original_image_pil = Image.open(uploaded_file)
        original_image_pil = resize_image(original_image_pil)
        
        channels, processed_pil = get_image_channels(original_image_pil, is_grayscale)
        original_image_np = np.array(processed_pil, dtype=np.float32)
        
        height, width = channels[0].shape
        
        # Perform SVD
        U_all, s_all, Vh_all = perform_svd(channels)
        
        max_k = len(s_all[0])
        
        k = st.slider(
            "Number of Singular Values (k)", 
            min_value=1, 
            max_value=max_k, 
            value=min(30, max_k),
            help="Select how many of the 'most important' singular values to use for reconstruction."
        )

# --- Main Area for Display ---
if uploaded_file:
    # Reconstruct images using the selected k
    reconstructed_image_pil, least_important_image_pil = reconstruct_images(U_all, s_all, Vh_all, k, is_grayscale)
    reconstructed_image_np = np.array(reconstructed_image_pil)

    # Calculate metrics
    metrics = calculate_metrics(height, width, len(channels), U_all, s_all, Vh_all, k)

    st.markdown("---")
    
    # --- Display Images and Metrics ---
    st.subheader("🖼️ Image Reconstruction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image_pil, caption=f"Original Image ({width}x{height})", use_container_width=True)
        st.download_button("Download Original", image_to_bytes(original_image_pil), "original.png")

    with col2:
        st.image(reconstructed_image_pil, caption=f"Reconstructed (k={k})", use_container_width=True)
        st.info("This image is reconstructed using only the top `k` singular values.", icon="💡")
        st.metric(label="Compression Ratio", value=f"{metrics['compression_factor']:.1f}x")
        st.download_button("Download Reconstructed", image_to_bytes(reconstructed_image_pil), f"reconstructed_k{k}.png")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🔍 Visualizing Rank Importance")
    col3, col4 = st.columns(2)

    with col3:
        st.image(least_important_image_pil, caption=f"Least Important Ranks (k={k})", use_container_width=True)
        st.info("This image is built from the `k` smallest singular values.", icon="💡")
        st.download_button("Download Least Important", image_to_bytes(least_important_image_pil), f"least_important_k{k}.png")

    with col4:
        fig_diff = plot_difference_heatmap(original_image_np, reconstructed_image_np, is_grayscale)
        st.pyplot(fig_diff, use_container_width=True)
        st.info("The heatmap shows information loss. Brighter areas mean greater difference.", icon="💡")

    st.markdown("---")

    # --- Animated Reconstruction ---
    st.subheader("🎥 Animated Reconstruction")
    if st.button("▶️ Play Animation (1 to k)", help=f"Reconstructs the image from k=1 up to k={k}"):
        anim_placeholder = st.empty()
        status_text = st.empty()
        for i in range(1, k + 1):
            anim_img, _ = reconstruct_images(U_all, s_all, Vh_all, i, is_grayscale)
            anim_placeholder.image(anim_img, caption=f"Reconstructing with k={i}", use_container_width=True)
            status_text.progress(i / k)
            time.sleep(0.05)
        status_text.success(f"Animation finished at k={k}!")

    st.markdown("---")

    # --- Plots and Data ---
    st.subheader("📊 Analysis & Metrics")
    
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Singular Value Plots")
        fig_sv = plot_singular_values(s_all, is_grayscale)
        st.pyplot(fig_sv)
    with col6:
        st.markdown("#### Cumulative Energy Plot")
        fig_energy = plot_cumulative_energy(s_all, is_grayscale)
        st.pyplot(fig_energy)

    st.markdown("#### SVD Compression Report")
    
    # New layout for clarity
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Original Data Size", format_bytes(metrics['original_storage_bytes']))
    with col_b:
        st.metric("Compressed SVD Size", format_bytes(metrics['compressed_storage_bytes']))

    metric1, metric2 = st.columns(2)
    metric1.metric("Data Size Reduction", f"{metrics['storage_savings']:.2f}%")
    metric2.metric("Compression Ratio", f"{metrics['compression_factor']:.1f}x")

    st.info(
        "**Note:** These sizes reflect the theoretical data needed for SVD, not the final downloaded PNG file size, which includes its own overhead and compression.",
        icon="💡"
    )

    if metrics['storage_savings'] < 0:
        st.warning(
            f"⚠️ With k={k}, the SVD representation requires {abs(metrics['storage_savings']):.2f}% more storage than the original."
        )

    st.markdown("---")

# --- SVD Explanation ---
display_svd_explanation()
