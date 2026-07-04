# 🏡 House Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project predicts **California house prices** using **Machine Learning**. It demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, model training, cross-validation, and prediction.

The project uses **Random Forest Regression** along with **Scikit-learn Pipelines** to build a reusable and production-ready ML pipeline.

---

## 🚀 Features

- Data preprocessing using Scikit-learn Pipelines
- Missing value handling with `SimpleImputer`
- Feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Stratified train-test split
- Model training using Random Forest Regressor
- 5-Fold Cross Validation
- Model serialization using Joblib
- Separate training and prediction scripts
- Clean project structure

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   └── housing.csv
│
├── train.py
├── predict.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📊 Machine Learning Workflow

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ├── Missing Value Imputation
   ├── Feature Scaling
   └── One-Hot Encoding
   │
   ▼
Train-Test Split
   │
   ▼
Random Forest Regressor
   │
   ▼
5-Fold Cross Validation
   │
   ▼
Model Evaluation
   │
   ▼
Save Model (.pkl)
   │
   ▼
Prediction on New Data
```

---

## 📁 Dataset

The project uses the **California Housing Dataset**.

Features include:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

**Target Variable**

- Median House Value

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Hemant84498/House-Price_Prediction.git
```

Move into the project directory

```bash
cd House-Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This will:

- Load the dataset
- Preprocess the data
- Train the Random Forest model
- Perform 5-Fold Cross Validation
- Save the trained model

---

## 🔮 Make Predictions

```bash
python predict.py
```

This loads the saved model and generates predictions on the input dataset.

---

## 📈 Model Evaluation

The model is evaluated using:

- Root Mean Squared Error (RMSE)
- 5-Fold Cross Validation

These metrics help evaluate model performance and generalization.

---

## 💡 Future Improvements

- Hyperparameter tuning using GridSearchCV
- XGBoost Regressor
- LightGBM
- CatBoost
- Feature Importance Visualization
- Model Deployment using Flask/FastAPI
- Docker Containerization
- CI/CD using GitHub Actions

---

## 👨‍💻 Author

**Hemant**

Engineering Student | Python Developer | Aspiring Data Scientist

GitHub: https://github.com/Hemant84498

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.

It helps support the project and motivates future improvements.

---

## 📄 License

This project is licensed under the MIT License.