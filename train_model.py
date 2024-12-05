import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Load dataset
data = pd.read_csv("dataset/final_df.csv")

# Konversi gender ke numerik
data["gender"] = data["gender"].map({"Male": 0, "Female": 1})

# Fitur input: Gender, BMI, Age
X = data[["gender", "bmi", "age"]]

# Target output: Daily Steps
y = data[["daily_steps"]]

# Standarisasi data
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)

# Simpan scaler
joblib.dump(scaler_X, "scaler/scaler_X.pkl")
joblib.dump(scaler_y, "scaler/scaler_y.pkl")

# Standarisasi data input dan output
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# Membangun model sederhana
model = Sequential([
    Dense(64, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Latih model
model.fit(X_scaled, y_scaled, epochs=50, batch_size=8, verbose=1)

# Simpan model
model.save("model/model1.h5")

print("Model dan scaler berhasil disimpan!")
