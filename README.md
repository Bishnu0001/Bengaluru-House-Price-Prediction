# 🏠 Bengaluru House Price Prediction

This project predicts house prices in Bengaluru using Machine Learning (Linear Regression). It includes data preprocessing, feature engineering, correlation-based feature selection, and model evaluation.

---

## 📌 Features
- Data Cleaning and Preprocessing
- Handling missing values
- Conversion of `total_sqft` values (range to average)
- One-Hot Encoding for categorical features
- Correlation-based Feature Selection
- Linear Regression Model
- Model Evaluation using RMSE and R² Score

---

## 📂 Dataset
- File: `bengaluru_house_prices.csv`
- Contains housing data such as location, size, total_sqft, bath, price, etc.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 How It Works

### 1. Data Preprocessing
- Convert `total_sqft` values into numeric format
- Handle ranges like `1200-1500`
- Remove missing values

### 2. Feature Engineering
- Apply One-Hot Encoding using `pd.get_dummies()`
- Drop unnecessary columns and NaN values

### 3. Feature Selection
- Select features with correlation ≥ 0.5 with price

### 4. Model Training
- Train a Linear Regression model
- Split dataset into training and testing (75:25)

### 5. Evaluation
- R² Score (Accuracy)
- RMSE (Root Mean Squared Error)


