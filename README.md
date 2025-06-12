# Fraud Detection Using Hybrid Model (XGBoost + Autoencoder)

This project implements a hybrid machine learning approach combining **XGBoost** and **Autoencoder** to detect fraudulent transactions in a credit card dataset. It features a user-friendly web interface built with **Flask** that allows users to input transaction details and receive instant predictions.

---

## 🔍 Overview

- **Problem**: Detecting fraudulent transactions among millions of legitimate ones.
- **Solution**: A hybrid model that leverages both supervised (XGBoost) and unsupervised (Autoencoder) learning.
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)

---

## 🧠 Model Architecture

- **Autoencoder**:
  - Learns compressed representations of normal transactions.
  - Used for anomaly detection.
- **XGBoost**:
  - Trained on labeled data.
  - Provides powerful classification capabilities.

Combined results provide improved accuracy and reduced false positives.

---

## 💻 How to Use

### 1. Clone the repository:
```bash
git clone https://github.com/VanguriMadhulika/fraud-detection-hybrid.git
cd fraud-detection-hybrid
2. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
3. Run the web application:
bash
Copy
Edit
python app.py
4. Open the app in your browser:
Go to: http://127.0.0.1:5000

📝 Features
✅ Real-time prediction via web form.

✅ Hybrid model with increased accuracy.

✅ Clear output: "Fraud" or "Not Fraud"

✅ Lightweight and easy to deploy.

⚠️ Note
The original dataset creditcard.csv exceeds GitHub's upload limit and is not included here. You can download it from:

🔗 Kaggle - Credit Card Fraud Detection Dataset

Place it inside a data/ folder if required for retraining.

📊 Future Improvements
Add LSTM-based detection.

Build a dashboard for model analytics.

Deploy on cloud (e.g., AWS, Heroku).



