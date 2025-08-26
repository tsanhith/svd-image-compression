# 🖼️ SVD Image Explorer: An Interactive Deep Dive into Image Compression

**SVD Image Explorer** is a high-performance Streamlit web application that **demonstrates image compression using Singular Value Decomposition (SVD)**. Users can upload images and instantly see the trade-offs between data compression and visual fidelity.

---

## ✨ Core Features

- **Real-time SVD Computation:** Upload JPG, JPEG, or PNG images and compute SVD instantly.  
- **Interactive Rank Selection:** Slider to select the number of singular values (`k`) for reconstruction.  
- **Side-by-Side Comparison:** Original vs reconstructed image visualization.  
- **Least Important Ranks Visualization:** See discarded image details using the last singular values.  
- **Information Loss Heatmap:** Visualize absolute error between original and reconstructed images.  
- **Animated Reconstruction:** Gradually build the image from rank 1 to the selected rank for a dynamic understanding.  
- **Singular Value & Cumulative Energy Plots:** Visualize energy distribution across singular values for each channel.  
- **Quantitative Compression Report:** Metrics including SVD values stored, compression ratio, storage savings, and pixel counts.

---

## 🔬 Most vs. Least Important Details

This shows the power of SVD:

- **Most Important Ranks (Top k):** Capture the main structure and shapes of the image.  
- **Least Important Ranks (Last k):** Capture subtle textures, fine details, and noise.  
- Helps users understand which parts of the image contribute most to its perceptual quality.

---

## 📊 Feature Workflow / Roadmap

```
User Uploads Image
        │
        ▼
  Pre-processing (Resize & Convert to Float32)
        │
        ▼
      SVD Computation
        │
        ├──> Singular Value & Cumulative Energy Plot
        │
        ├──> Animated Reconstruction (k=1 → selected k)
        │
        ├──> Image Reconstruction:
        │       ├─ Most Important Ranks (Top k)
        │       └─ Least Important Ranks (Last k)
        │
        └──> Information Loss Heatmap
        │
        ▼
   Metrics & Compression Report
```

This workflow helps users understand how SVD deconstructs and reconstructs image information.

---

## 🛠️ Technical Deep Dive & Optimizations

### Performance & Memory Management
- **Pre-computation Resizing:** Uploaded images are automatically resized to a maximum dimension of 1024px to reduce computation while preserving visual quality.  
- **Efficient SVD Computation:** Uses `numpy.linalg.svd(full_matrices=False)` for thin SVD, reducing memory usage and speeding up computations.  
- **Optimized Data Types:** Images are converted to `np.float32` before SVD for numerical stability and compatibility with linear algebra routines.  

### Caching Strategy (Future Enhancement)
- SVD computations can be cached using `@st.cache_data` to prevent repeated processing, improving performance for repeated runs or multiple users.  

```python
@st.cache_data
def perform_svd(channels):
    # ... SVD logic ...
```

### Modular & Scalable Architecture
- **app.py:** Handles UI layout and user interactions.  
- **src/image_processing.py:** Core numerical and image processing logic.  
- **src/plotting.py:** Visualization and plotting logic.  
- **src/utils.py:** Helper functions (byte formatting, image resizing, etc.).  

This modular structure ensures easy maintenance, scalability, and separation of concerns.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+  
- pip and venv  

### Setup Instructions
1. Clone the repository:
```bash
git clone <your-repository-url>
cd svd-image-explorer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

---

## ⚠️ Notes on SVD vs File Size

- **SVD Compression Report:** Shows **theoretical size** based on number of floating-point values required for SVD.  
- **PNG File Size:** May differ due to PNG's internal lossless compression.  
- High-rank reconstructions may sometimes result in PNG files larger than lower-rank reconstructions due to pattern efficiency.

---

## 📊 Learning Insights

- **Energy Preserved Metric:** Shows percentage of total image energy retained by top k singular values.  
- **Difference Visualization:** Highlights areas where image information is lost.  
- **Animated Reconstruction:** Builds image rank-by-rank to demonstrate reconstruction visually.  
- **Cumulative Energy Plot:** Emphasizes that most image energy is concentrated in top singular values.

---

## 🖥️ Technical Stack

- Python 3.8+  
- Streamlit  
- NumPy  
- OpenCV  
- Pillow  
- Matplotlib  
- SciPy  

---

## 📄 File Structure

```
svd-image-explorer/
│
├─ app.py                  # Main Streamlit app
├─ requirements.txt        # Python dependencies
└─ src/
   ├─ image_processing.py  # SVD computation and image manipulation
   ├─ plotting.py          # Visualization functions
   └─ utils.py             # General helper functions
```

---

## 📚 References

- [Singular Value Decomposition - Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)  
- [NumPy.linalg.svd Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)  
- [Streamlit Docs](https://docs.streamlit.io/)
