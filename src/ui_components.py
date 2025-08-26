import streamlit as st

def display_svd_explanation():
    """
    Displays the SVD explanation in a Streamlit expander.
    """
    with st.expander("What is Singular Value Decomposition (SVD)?"):
        st.markdown("""
        **Singular Value Decomposition (SVD)** is a fundamental matrix factorization technique in linear algebra. It states that any matrix $A$ (like an image) can be broken down into three other matrices:

        $$ A = U \\Sigma V^T $$

        - **$U$**: An orthogonal matrix whose columns are the *left-singular vectors*. For an image, these vectors represent abstract features related to the rows (height).
        - **$\\Sigma$ (Sigma)**: A diagonal matrix containing the **singular values** ($s_i$). These values are non-negative and are sorted in descending order ($s_1 \\geq s_2 \\geq \\dots \\geq 0$). They represent the "importance" or "energy" of each component.
        - **$V^T$**: An orthogonal matrix whose rows are the *right-singular vectors*. For an image, these vectors represent abstract features related to the columns (width).
        
        #### How does this apply to image compression?
        The key idea is that the first few singular values in $\\Sigma$ are much larger than the rest. These large values correspond to the most significant, low-frequency features of the image—like its overall structure, shapes, and colors. The smaller singular values correspond to high-frequency details, like fine textures and noise.
        
        By keeping only the top **$k$** singular values (and the corresponding vectors in $U$ and $V^T$), we can create a very good approximation of the original image:
        
        $$ A_k = U_k \\Sigma_k V_k^T $$
        
        Here, $A_k$ is the **reconstructed image**, which requires storing far less data than the original image $A$. This app lets you control `k` to see the trade-off between image quality and data storage.
        """)
