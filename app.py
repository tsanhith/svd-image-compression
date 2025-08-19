import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="SVD Image Compression Showcase",
    page_icon="🖼️"
)


# --- Helper Function for Downloads ---
def image_to_bytes(img_array):
    """Converts a NumPy image array to bytes for downloading."""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if not is_success:
        return None
    return io.BytesIO(buffer).getvalue()


# --- Caching and Core Functions ---

@st.cache_data
def get_svd_components(img):
    """
    Performs SVD on each color channel of the image and returns all components.
    """
    components = {'R': None, 'G': None, 'B': None}
    for i, channel in enumerate(['R', 'G', 'B']):
        U, s, Vt = np.linalg.svd(img[:, :, i], full_matrices=False)
        components[channel] = {'U': U, 's': s, 'Vt': Vt}
    return components

@st.cache_data
def svd_reconstruct(svd_components, k, use_last_k=False):
    """
    Reconstructs an image from SVD components using either the first k or last k singular values.
    """
    reconstructed_channels = []
    for channel in ['R', 'G', 'B']:
        U = svd_components[channel]['U']
        s = svd_components[channel]['s']
        Vt = svd_components[channel]['Vt']
        
        if use_last_k:
            # Use the least significant singular values
            reconstructed_matrix = U[:, -k:] @ np.diag(s[-k:]) @ Vt[-k:, :]
        else:
            # Use the most significant singular values
            reconstructed_matrix = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            
        reconstructed_channels.append(reconstructed_matrix)

    reconstructed_img = np.stack(reconstructed_channels, axis=2)
    
    # Normalize for visualization if it's the least important ranks
    if use_last_k:
        if reconstructed_img.max() != reconstructed_img.min():
            reconstructed_img = 255 * (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
    
    return reconstructed_img.clip(0, 255).astype(np.uint8)


@st.cache_data
def create_singular_value_plot(svd_components):
    """
    Creates a matplotlib plot to visualize the magnitude of singular values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for channel, comps in svd_components.items():
        ax.plot(range(1, len(comps['s']) + 1), comps['s'], label=f'{channel} channel')
    ax.set_title("Singular Values Distribution", fontsize=16)
    ax.set_xlabel("Singular Value Index (Rank)", fontsize=12)
    ax.set_ylabel("Magnitude (Log Scale)", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    return fig


# --- UI Layout ---

st.title("🖼️ SVD Image Compression Showcase")
st.markdown(
    "Explore how **Singular Value Decomposition (SVD)** intelligently deconstructs images. "
    "This app focuses purely on SVD to show how it separates important vs. unimportant details."
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    use_grayscale = st.checkbox("Convert to Grayscale", help="Analyze SVD on a single 2D matrix.")

    # Placeholder for the slider
    k_slider_placeholder = st.empty()

if uploaded_file is not None:
    # --- Image Loading and Processing ---
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if use_grayscale:
            gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            # Duplicate the grayscale channel to make it a 3-channel image for consistent processing
            img_rgb = np.stack([gray_img]*3, axis=-1)

        original_pixel_count = img_rgb.shape[0] * img_rgb.shape[1] * (1 if use_grayscale else 3)
        max_k = min(img_rgb.shape[:2])
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # --- Add the slider to the sidebar ---
    with k_slider_placeholder.container():
        k = st.slider(
            "Number of Ranks (k)",
            min_value=1,
            max_value=max_k,
            value=min(50, max_k),
            step=1,
            help="The number of singular values (ranks) to use for reconstruction."
        )

    # --- Perform SVD ---
    svd_components = get_svd_components(img_rgb)
    
    # --- Reconstruct Images ---
    compressed_img = svd_reconstruct(svd_components, k)
    
    # --- Main Content Area ---
    st.header("Primary Reconstruction")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, use_column_width=True)
        st.info(f"**Total Pixel Values:** {original_pixel_count:,}")
        st.download_button(
           label="Download Original Image",
           data=image_to_bytes(img_rgb),
           file_name="original.png",
           mime="image/png"
        )

    with col2:
        st.subheader("SVD Reconstructed Image")
        st.image(compressed_img, use_column_width=True, caption=f"Reconstructed with the top {k} singular values")
        
        # Calculate data size for SVD components
        m, n = img_rgb.shape[0], img_rgb.shape[1]
        svd_values_count = (m * k + k + k * n) * (1 if use_grayscale else 3)
        
        # Calculate metrics
        compression_ratio = original_pixel_count / svd_values_count if svd_values_count > 0 else float('inf')
        storage_saving = (1 - (svd_values_count / original_pixel_count)) * 100 if original_pixel_count > 0 else 0

        st.info(f"**SVD Values Stored:** {int(svd_values_count):,}")
        st.metric(label="Compression Ratio", value=f"{compression_ratio:.1f}x")
        st.metric(label="Storage Savings", value=f"{storage_saving:.1f}%")

        st.download_button(
           label="Download Reconstructed Image",
           data=image_to_bytes(compressed_img),
           file_name=f"reconstructed_k{k}.png",
           mime="image/png"
        )


    # --- NEW: Section to Compare Most vs. Least Important Ranks ---
    st.write("---")
    st.header("🔬 Most vs. Least Important Details")
    st.markdown("This shows the power of SVD. The image on the left is built from the **most significant** `k` ranks. The image on the right is built from the **least significant** `k` ranks.")

    least_important_img = svd_reconstruct(svd_components, k, use_last_k=True)
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.image(compressed_img, caption=f"Image from First {k} Ranks (Most Important)", use_column_width=True)
    with comp_col2:
        st.image(least_important_img, caption=f"Image from Last {k} Ranks (Least Important)", use_column_width=True)
        st.download_button(
           label="Download 'Least Important' Image",
           data=image_to_bytes(least_important_img),
           file_name=f"least_important_k{k}.png",
           mime="image/png"
        )


    # --- Explanations and Plots ---
    st.write("---")
    st.header("🔍 Deeper Dive into SVD")

    with st.expander("📊 See Singular Value Plot"):
        fig = create_singular_value_plot(svd_components)
        st.pyplot(fig)
        st.markdown(
            "This plot shows the magnitude of each singular value, sorted from largest to smallest. "
            "Notice the sharp drop-off. This confirms that most of the image's 'energy' is concentrated "
            "in the first few values, which is why they are the most important."
        )

    with st.expander("🧠 Learn about the technique"):
        st.markdown(r"""
        **Singular Value Decomposition (SVD)** is a powerful matrix factorization technique that separates a matrix into layers of decreasing importance.

        1.  **Decomposition**: An image's matrix ($A$) is broken into three other matrices: $U$, $\Sigma$ (Sigma), and $V^T$.
            $$ A = U \cdot \Sigma \cdot V^T $$
        2.  **Hierarchy of Importance**: The $\Sigma$ matrix contains the singular values on its diagonal, sorted by magnitude from top-left to bottom-right. The first few values are the largest and correspond to the most significant features. **The least important singular values are the last ones on this diagonal.**
        3.  **Reconstruction**: By selecting only the top **`k`** singular values, we create a "low-rank approximation" that captures the essence of the image with far less data.
            $$ A_{\text{reconstructed}} \approx U_{:, :k} \cdot \Sigma_{:k, :k} \cdot V^T_{:k, :} $$
        This app demonstrates that the information stored in the last singular values is visually much less important than the information in the first few.
        """)

else:
    # Initial message to guide the user
    st.info("Upload an image using the sidebar to begin.")
