from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.decomposition import PCA

# Muat model yang telah dilatih
model = tf.keras.models.load_model('PCA_model.h5')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Halaman beranda
@app.route('/')
def home():
    return render_template('home.html')

# Halaman klasifikasi gambar
@app.route('/classify', methods=['POST'])
def classify():
    # Ambil gambar yang diunggah
    image = request.files['image']

    image = Image.open(image)

    # Ubah ukuran gambar baru menjadi 32x32 piksel
    new_image = image.resize((32, 32))

    # Konversi gambar baru menjadi array NumPy
    new_image = np.array(new_image)

    # Ratakan gambar baru
    new_image_flat = new_image.reshape(1, -1)

    # Terapkan PCA
    pca = PCA(0.9)
    pca.fit(new_image_flat)

    new_image_flat = new_image.reshape(1, pca.n_components_)

    new_image_pca = pca.transform(new_image_flat)

    # Prediksi label kelas
    prediction = model.predict(new_image_pca)

    # Dapatkan label kelas yang diprediksi
    labels = '''moi non_moi'''.split()
    predicted_class = np.argmax(prediction)
    predicted_label = labels[np.argmax(predicted_class)]

    # Render halaman hasil
    return render_template('result.html', prediction=predicted_label)


# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)