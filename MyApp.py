import streamlit as st
import pandas as pd
import pca5_modif as pcas
from PIL import Image

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    new_image = Image.open(uploaded_file)
    new_image = new_image.resize((700, 700))

    # Prediksi dan tampilkan gambar dengan PCA
    image_pca, prediction = pcas.predict_image(uploaded_file)
    image_pca = Image.fromarray(image_pca)  # Konversi array NumPy menjadi objek gambar

    # Konversi objek gambar menjadi array NumPy)
    image_pca = image_pca.resize((700, 700))

    # uploaded_file = uploaded_file.resize((700, 700))

    col1, col2= st.columns(2)

    with col1:
        st.header("Tanpa PCA")
        st.image(new_image)

    with col2:
        st.header("Dengan PCA")
        st.image(image_pca, use_column_width=True)

    st.title("Hasil prediksi")
    st.subheader(prediction)