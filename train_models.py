import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features and label
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
X = df[features]
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Train Autoencoder --------------------
print("Training Autoencoder...")

input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(14, activation='relu')(input_layer)
encoded = Dense(7, activation='relu')(encoded)
decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

X_train_auto, _, = train_test_split(X_scaled[y == 0], test_size=0.2, random_state=42)

autoencoder.fit(
    X_train_auto,
    X_train_auto,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1
)

# Save autoencoder
autoencoder.save("models/autoencoder_model.keras")

# -------------------- Train XGBoost --------------------
print("Training XGBoost...")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Save XGBoost model
joblib.dump(xgb, "models/xgboost_model.json")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models and scaler saved successfully in the 'models/' directory.")