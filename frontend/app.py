import io
import requests
import streamlit as st
import cv2
import os
import numpy as np
import time

url = "http://localhost:8000"
url_add_noise = url + "/add_noise"
url_remove_noise = url + "/remove_noise"
url_sharpen_image = url + "/sharpen_image"


# Fungsi untuk mendeteksi wajah

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

# Fungsi untuk menambahkan noise salt and pepper

def add_salt_and_pepper_noise(image, prob):
    output = np.copy(image)
    black = 0
    white = 255
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def process_image_in_fastapi(image_response):
    image = np.frombuffer(image_response, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def process_api_add_noise(image_path, noise_probability):
    # Buka gambar dari path sebagai file binary
    with open(image_path, "rb") as image_file:
        # Kirimkan gambar ke API FastAPI
        response = requests.post(
            url_add_noise,
            files={"file": image_file},
            data={"noise_prob": noise_probability}
        )
        
    if response.status_code == 200:
        # Konversi konten respons menjadi numpy array
        return process_image_in_fastapi(response.content)
    else:
        st.error(f"Terjadi kesalahan saat memproses gambar: {response.status_code}")
        return None

# API untuk menghilangkan noise
def process_api_remove_noise(noisy_image):
    # Encode gambar numpy array ke format biner (misalnya .png atau .jpg)
    _, image_encoded = cv2.imencode('.png', noisy_image)
    
    # Konversi gambar menjadi file-like object untuk dikirimkan
    image_file = io.BytesIO(image_encoded.tobytes())

    # Kirimkan gambar yang sudah di-encode ke API FastAPI
    response = requests.post(
        url_remove_noise,
        files={"file": ("image.png", image_file, "image/png")}
    )
        
    if response.status_code == 200:
        # Konversi konten respons menjadi numpy array
        return process_image_in_fastapi(response.content)
    else:
        st.error(f"Terjadi kesalahan saat memproses gambar: {response.status_code}")
        return None
    
# API untuk mempertajam gambar
def process_api_sharpen_image(denoised_image):
    # Encode gambar numpy array ke format biner (misalnya .png atau .jpg)
    _, image_encoded = cv2.imencode('.png', denoised_image)
    
    # Konversi gambar menjadi file-like object untuk dikirimkan
    image_file = io.BytesIO(image_encoded.tobytes())

    # Kirimkan gambar yang sudah di-encode ke API FastAPI
    response = requests.post(
        url_sharpen_image,
        files={"file": ("image.png", image_file, "image/png")}
    )
        
    if response.status_code == 200:
        # Konversi konten respons menjadi numpy array
        return process_image_in_fastapi(response.content)
    else:
        st.error(f"Terjadi kesalahan saat memproses gambar: {response.status_code}")
        return None

# Fungsi untuk menghilangkan noise menggunakan median filtering (dinonaktifkan)

# def remove_noise(image):
#     return cv2.medianBlur(image, 5)

# Fungsi untuk mempertajam gambar

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# Aplikasi Streamlit
st.title("Tambah Wajah Baru ke Dataset")

# Input untuk nama orang baru
new_person = st.text_input("Masukkan nama orang baru:")

# Tombol untuk menangkap gambar wajah baru
capture = st.button("Tambahkan Wajah Baru")

if capture:
    if not new_person:
        st.warning("Silakan masukkan nama orang baru.")
    else:
        save_path = os.path.join('dataset', new_person)

        if not os.path.exists('dataset'):
            os.makedirs('dataset')
            st.info("Folder 'dataset' telah dibuat.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            st.success(f"Folder untuk {new_person} telah dibuat.")

            # Mulai menangkap gambar dari webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error(
                    "Tidak dapat membuka webcam. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain.")
            else:
                num_images = 0
                max_images = 20  # Tangkap 20 gambar wajah

                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    while num_images < max_images:
                        ret, frame = cap.read()
                        if not ret:
                            st.error(
                                "Error: Tidak dapat membaca frame dari webcam.")
                            break

                        # Deteksi wajah dalam frame
                        faces = detect_faces(frame)

                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                face = frame[y:y+h, x:x+w]
                                img_name = os.path.join(
                                    save_path, f"img_{num_images}.jpg")
                                cv2.imwrite(img_name, face)
                                num_images += 1

                                # Gambarkan kotak di sekitar wajah yang terdeteksi
                                cv2.rectangle(
                                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                                # Tampilkan hasil deteksi
                                frame_rgb = cv2.cvtColor(
                                    frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(
                                    frame_rgb, channels="RGB", caption=f"Gambar {num_images}/{max_images}")

                                # Perbarui progress bar
                                progress = num_images / max_images
                                progress_bar.progress(progress)
                                status_text.text(
                                    f"Menyimpan gambar {num_images} dari {max_images}...")

                                break
                        else:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(
                                frame_rgb, channels="RGB", caption="Tidak ada wajah terdeteksi.")

                        time.sleep(0.1)

                    st.success(
                        f"{num_images} gambar telah berhasil ditambahkan ke dataset {new_person}.")
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    progress_bar.empty()
                    status_text.empty()
        else:
            st.warning(
                "Nama sudah ada di dataset. Silakan pilih nama lain atau tambahkan lebih banyak gambar.")

# Opsi pengolahan citra
st.header("Pengolahan Citra Wajah")

# Slider untuk mengatur lebar tampilan gambar
image_width = st.slider("Atur ukuran lebar gambar:",
                        min_value=100, max_value=1000, value=200)

# Secara otomatis menampilkan semua gambar dalam direktori dataset untuk dipilih
dataset_folder = 'dataset'
if os.path.exists(dataset_folder):
    all_images = []
    # Cari semua gambar di direktori dataset
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))

    if all_images:
        # Pilih gambar dari dropdown
        selected_image_path = st.selectbox(
            "Pilih citra untuk diolah:", all_images)

        # Pratinjau gambar yang dipilih
        if selected_image_path:
            image = cv2.imread(selected_image_path)
            st.image(image, caption="Preview Citra Terpilih",
                     channels="BGR", width=image_width)

        # Slider untuk mengatur probabilitas noise
        noise_probability = st.slider(
            "Atur tingkat noise salt & pepper:", min_value=0.0, max_value=1.0, value=0.05)
        
        # Tombol untuk memproses gambar yang dipilih
        process = st.button("Proses Citra")

        if noise_probability:
            # Tambahkan noise salt and pepper dengan probabilitas yang dapat disesuaikan
            noisy_image = process_api_add_noise(selected_image_path, noise_probability)

            # Hilangkan noise
            denoised_image = process_api_remove_noise(noisy_image)

            # Pertajam gambar
            sharpened_image = process_api_sharpen_image(denoised_image)

            cols = st.columns(3)
            with cols[0]:
                st.image(noisy_image, caption="Citra dengan Noise Salt & Pepper",
                         channels="BGR", width=image_width)
            with cols[1]:
                st.image(denoised_image, caption="Citra Setelah Menghilangkan Noise",
                         channels="BGR", width=image_width)
            with cols[2]:
                st.image(sharpened_image, caption="Citra Setelah Penajaman",
                         channels="BGR", width=image_width)

            folder_name = st.text_input("Masukan nama folder penyimpanan:")
            save_processed = st.button("Simpan Citra yang Diproses")

            if save_processed:
                if not folder_name:
                    folder_name = "temp"
                # Simpan gambar yang sudah diproses
                processed_path = os.path.join('processed_dataset', new_person)
                if not os.path.exists(processed_path):
                    os.makedirs(processed_path)

                cv2.imwrite(os.path.join(processed_path,
                            'noisy_image.jpg'), noisy_image)
                cv2.imwrite(os.path.join(processed_path,
                            'denoised_image.jpg'), denoised_image)
                cv2.imwrite(os.path.join(processed_path,
                            'sharpened_image.jpg'), sharpened_image)
                st.success(
                    f"Citra berhasil diproses dan disimpan di folder processed_dataset/{folder_name}.")
    else:
        st.warning(
            "Tidak ada gambar dalam dataset. Silakan tambahkan wajah baru terlebih dahulu.")
else:
    st.warning(
        "Folder dataset tidak ditemukan. Silakan tambahkan wajah baru terlebih dahulu.")
