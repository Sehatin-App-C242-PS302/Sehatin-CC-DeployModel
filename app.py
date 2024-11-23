from string import ascii_uppercase, digits

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import cv2
import keras
import h5py
from keras.models import load_model
import uvicorn

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Load model
# Proses ini memuat file model terlatih "model1.h5"
# Pastikan file "model1.h5" berada di direktori "model" atau sesuaikan path-nya
model_path = "model/model1.h5"
model = load_model(model_path)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sehatin - BMI Calculator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8fbf7;
                margin: 0;
                padding: 20px;
            }
            h1 {
                text-align: center;
                color: #3a5f5c;
            }
            form {
                max-width: 400px;
                margin: 0 auto;
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            label {
                font-weight: bold;
                margin-bottom: 5px;
                display: block;
                color: #3a5f5c;
            }
            input, select, button {
                width: 100%;
                padding: 10px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                background-color: #3a5f5c;
                color: white;
                font-size: 16px;
                cursor: pointer;
                border: none;
            }
            button:hover {
                background-color: #2d4643;
            }
        </style>
    </head>
    <body>
        <h1>Masukkan Data Kamu Yuk</h1>
        <form action="/calculate_bmi" method="post">
            <label for="gender">Gender:</label>
            <select name="gender" id="gender" required>
                <option value="" disabled selected>Pilih Gender</option>
                <option value="male">Laki-laki</option>
                <option value="female">Perempuan</option>
            </select>
            <label for="age">Age:</label>
            <input type="number" name="age" id="age" min="1" required>
            <label for="height">Height (cm):</label>
            <input type="number" name="height" id="height" min="1" required>
            <label for="weight">Weight (kg):</label>
            <input type="number" name="weight" id="weight" min="1" required>
            <button type="submit">Count BMI</button>
        </form>
    </body>
    </html>
    """


@app.post("/calculate_bmi")
@app.post("/calculate_bmi")
async def calculate_bmi(
    gender: str = Form(...),
    age: int = Form(...),
    height: float = Form(...),
    weight: float = Form(...)
):
    try:
        # Menghitung BMI
        height_in_meters = height / 100  # Konversi tinggi ke meter
        bmi = weight / (height_in_meters ** 2)

        # Klasifikasi BMI berdasarkan nilai
        if bmi < 18.5:
            category = "Underweight"
            daily_steps = "6,000–8,000 steps/day"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
            daily_steps = "8,000–10,000 steps/day"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
            daily_steps = "7,000–9,000 steps/day"
        else:
            category = "Obesity"
            daily_steps = "5,000–7,000 steps/day"

        # Mengembalikan hasil
        result = {
            "gender": gender.capitalize(),
            "age": age,
            "height_cm": height,
            "weight_kg": weight,
            "bmi": round(bmi, 2),
            "category": category,
            "daily_step_recommendation": daily_steps,
        }
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Buat label yang sesuai dengan output model
        label_names = digits + ascii_uppercase
        labels = {idx: label for idx, label in enumerate(label_names)}

        # Baca file gambar
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image_to_test = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess image
        gray = cv2.cvtColor(image_to_test, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
        resized = cv2.resize(gray, (28, 28))  # Ubah ukuran ke 28x28
        normalized = resized.astype("float32") / 255.0  # Normalisasi
        reshaped = np.expand_dims(normalized, axis=(0, -1))  # Ubah ke bentuk (1, 28, 28, 1)

        # Prediksi menggunakan model
        predictions = model.predict(reshaped)
        label_idx = np.argmax(predictions)
        label = labels[label_idx]

        # Format hasil prediksi
        result = {
            "prediction": label,
            "confidence": float(predictions[0][label_idx])
        }

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
