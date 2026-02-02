# 🏦 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Library-TensorFlow-orange)
![API](https://img.shields.io/badge/API-Scikit--Learn-yellow)

## 📌 Project Overview
The **Customer Churn Prediction System** is an end-to-end Machine Learning application designed to assist banking institutions in retaining customers. By analyzing demographic and financial data, the system predicts the likelihood of a customer leaving the bank (churning).

The application is built using **Artificial Neural Networks (ANN)** for high-accuracy predictions and features a user-friendly web interface powered by **Streamlit**.

---

## 📸 Interface

> The dashboard provides real-time risk assessment and actionable insights.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python-based Web Framework)
- **Backend/ML:** TensorFlow (Keras API), Scikit-Learn
- **Data Processing:** Pandas, NumPy

---

## 🧠 Model Architecture
The core "Brain" of the application is a Deep Learning model trained on the *Churn Modelling* dataset.
- **Input Layer:** 11 features (Credit Score, Geography, Gender, Age, Tenure, Balance, Products, Credit Card status, Active status, Salary).
- **Hidden Layers:** Two dense layers with 64 and 32 neurons, using `ReLU` activation.
- **Output Layer:** Single neuron with `Sigmoid` activation (Binary Classification).
- **Optimizer:** Adam.
- **Loss Function:** Binary Crossentropy.
- **Accuracy:** ~86% on unseen test data.

---

## 📂 Project Structure
```bash
Churn_Prediction_Project/
├── artifacts/              # Saved model files (.pkl and .h5)
│   ├── model.h5            # Trained Neural Network
│   ├── label_encoder.pkl   # Gender encoder
│   ├── one_hot_encoder.pkl # Geography encoder
│   └── scaler.pkl          # Feature Scaler
├── data/                   # Raw dataset (Churn_Modelling.csv)
├── notebooks/              # Jupyter Notebooks for experimentation
├── app.py                  # Main Streamlit Application
├── requirements.txt        # Python dependencies
└── README.md               # Project Documentation

```

---

## 📊 Dataset

The dataset used is `Churn_Modelling.csv`, containing 10,000 customer records.

* **Target Variable:** `Exited` (1 = Churn, 0 = Stay)
* **Key Features:** Credit Score, Geography, Gender, Age, Balance, etc.

---

