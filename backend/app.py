from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
import uvicorn
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Ganti dengan URL frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Fungsi untuk membaca gambar dari file yang diunggah
def read_image(uploaded_file):
    image_data = np.frombuffer(uploaded_file, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img

# Fungsi untuk mengubah gambar menjadi format yang bisa dikirimkan dalam response


def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    img_bytes = BytesIO(buffer)
    return img_bytes


# Fungsi untuk menerapkan konvolusi
def apply_convolution(image, kernel_type="average"):
    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        raise ValueError("Invalid convolution type")
    return cv2.filter2D(image, -1, kernel)

# Fungsi untuk menambahkan padding nol pada gambar


def apply_zero_padding(image, padding_size=10):
    return cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Fungsi untuk menerapkan filter (low, high, band pass)


def apply_filter(image, filter_type="low"):
    if filter_type == "low":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "band":
        low_pass = cv2.GaussianBlur(image, (9, 9), 0)
        high_pass = image - low_pass
        return low_pass + high_pass
    else:
        raise ValueError("Invalid filter type")

# Fungsi untuk menerapkan Transformasi Fourier


def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

# Fungsi untuk mengurangi noise periodik


def reduce_periodic_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius dari mask
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


# Praktikum streamlit dan opencv noise sharpening

def add_salt_and_pepper_noise(image, prob):
    output = np.copy(image)
    black = 0
    white = 255
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

# Function to remove noise using median filtering


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# Function to sharpen the image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    operation: str = Form(...),
    parameter: str = Form(None)  # Optional jika tidak diperlukan
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print(
        f"Processing image with operation: {operation} and parameter: {parameter}")

    # Cek jenis operasi yang diterima dari form-data
    if operation == "convolution":
        if not parameter:
            return {"error": "parameter is required for convolution"}
        processed_img = apply_convolution(img, parameter)

    elif operation == "zero_padding":
        if not parameter:
            return {"error": "parameter is required for zero_padding"}
        processed_img = apply_zero_padding(img, int(parameter))

    elif operation == "filter":
        if not parameter:
            return {"error": "parameter is required for filter"}
        processed_img = apply_filter(img, parameter)

    elif operation == "fourier_transform":
        processed_img = apply_fourier_transform(img)

    elif operation == "reduce_noise":
        processed_img = reduce_periodic_noise(img)

    else:
        return {"error": "Invalid operation"}

    _, buffer = cv2.imencode('.png', processed_img)
    byte_io = BytesIO(buffer)

    print(f"Image processed successfully with operation: {operation}")
    return StreamingResponse(byte_io, media_type="image/png")


# api streamlit opencv noise sharpening
@app.post("/add_noise/")
async def add_noise(
    file: UploadFile = File(...),
    noise_prob: float = Form(...)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print(
        f"Adding noise to image with and probability: {noise_prob}")

    # Cek jenis noise yang diterima dari form-data

    processed_img = add_salt_and_pepper_noise(img, noise_prob)
    
    _, buffer = cv2.imencode('.png', processed_img)
    byte_io = BytesIO(buffer)

    print(f"Noise added successfully with probability: {noise_prob}")
    return StreamingResponse(byte_io, media_type="image/png")



# remove noise
@app.post("/remove_noise/")
async def remove_noise_api(
    file: UploadFile = File(...)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print(
        f"Removing noise from image")

    processed_img = remove_noise(img)
    
    _, buffer = cv2.imencode('.png', processed_img)
    byte_io = BytesIO(buffer)

    print(f"Noise removed successfully")
    return StreamingResponse(byte_io, media_type="image/png")

# add sharpening
@app.post("/sharpen_image/")
async def sharpen_image_api(
    file: UploadFile = File(...)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print(
        f"Sharpening image")

    processed_img = sharpen_image(img)
    
    _, buffer = cv2.imencode('.png', processed_img)
    byte_io = BytesIO(buffer)

    print(f"Image sharpened successfully")
    return StreamingResponse(byte_io, media_type="image/png")










if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
