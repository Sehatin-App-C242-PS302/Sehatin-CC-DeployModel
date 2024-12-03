from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import uvicorn
import joblib
import logging
import pandas as pd

# Load dataset
data = pd.read_csv("dataset/final_df.csv")

# Logging setup
logging.basicConfig(level=logging.INFO)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Load model dan scaler
model_path = "model/model1.h5"
scaler_X_path = "scaler/scaler_X.pkl"
scaler_y_path = "scaler/scaler_y.pkl"

# Pastikan fungsi loss dikenali saat memuat model
model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})
scaler_X = joblib.load(scaler_X_path)  # Scaler untuk fitur input
scaler_y = joblib.load(scaler_y_path)  # Scaler untuk target output

# Fungsi untuk preprocessing input
def preprocess_input(gender: str, bmi: float, age: int):
    """Preprocessing input untuk model prediksi"""
    # Konversi gender dari string ke integer
    gender_numeric = 0 if gender.lower() == "male" else 1
    return np.array([[gender_numeric, bmi, age]])

@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sehatin - BMI Calculator</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f8fbf7; margin: 0; padding: 20px; }
            h1 { text-align: center; color: #3a5f5c; }
            form { max-width: 400px; margin: 0 auto; background: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); }
            label { font-weight: bold; margin-bottom: 5px; display: block; color: #3a5f5c; }
            input, select, button { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ccc; border-radius: 5px; }
            button { background-color: #3a5f5c; color: white; font-size: 16px; cursor: pointer; border: none; }
            button:hover { background-color: #2d4643; }
        </style>
    </head>
    <body>
        <h1>Masukkan Data Kamu Yuk</h1>
        <form action="/calculate_bmi" method="post">
            <label for="gender">Gender:</label>
            <select name="gender" id="gender" required>
                <option value="" disabled selected>Pilih Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
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
async def calculate_bmi(
    gender: str = Form(...),
    age: int = Form(...),
    height: float = Form(...),
    weight: float = Form(...)
):
    try:
        # Log input dari user
        logging.info(f"Input: Gender={gender}, Age={age}, Height={height}, Weight={weight}")

        # Hitung BMI
        height_in_meters = height / 100
        bmi = weight / (height_in_meters ** 2)
        logging.info(f"Calculated BMI: {bmi}")

        # Preprocessing input untuk model
        input_data = preprocess_input(gender, bmi, age)

        # Standardisasi input menggunakan scaler_X
        standardized_input = scaler_X.transform(input_data)
        logging.info(f"Standardized Input: {standardized_input}")

        # Prediksi daily steps
        daily_steps_scaled = model.predict(standardized_input)
        daily_steps = scaler_y.inverse_transform(daily_steps_scaled).flatten()[0]
        daily_steps = max(0, int(np.round(daily_steps / 100) * 100))
        logging.info(f"Predicted Daily Steps: {daily_steps}")

        # Klasifikasi BMI
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obesity"

        # Kembalikan hasil prediksi
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
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
