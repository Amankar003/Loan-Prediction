# Loan-Prediction

---
title: "Loan Prediction System"
author: "Aman Kumar"
description: "Machine Learning based system to predict loan approval based on applicant details."
tags: [machine learning, loan eligibility, classification, pandas, scikit-learn]
date: 2025-07-11
license: MIT
---

# 🏦 Loan Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

A machine learning-based loan eligibility prediction system. It predicts whether a loan application will be approved based on applicant details like income, employment, credit history, and more.

---

## 📌 Features

- Cleans and preprocesses applicant data  
- Handles missing values and encodes categorical variables  
- Uses Logistic Regression for prediction  
- Evaluates using accuracy, confusion matrix, classification report  
- Easily extendable for deployment

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`  
- **Algorithm:** Logistic Regression  
- **Environment:** Jupyter Notebook  

---

## 📂 Project Structure

```

Loan-Prediction/
│
├── Loan\_Prediction.ipynb         # Main notebook with model pipeline
├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

````

---

## 🧾 Input Features

| Feature           | Description                              |
|------------------|------------------------------------------|
| Gender           | Male / Female                            |
| Married          | Marital status                           |
| Education        | Graduate / Not Graduate                  |
| Self_Employed    | Employment status                        |
| ApplicantIncome  | Monthly income of applicant              |
| CoapplicantIncome| Monthly income of coapplicant (if any)   |
| LoanAmount       | Loan amount requested                    |
| Loan_Amount_Term | Loan term in months                      |
| Credit_History   | 1 (Yes), 0 (No)                          |
| Property_Area    | Urban / Rural / Semiurban                |

---

## 📈 Model Pipeline

1. Load & inspect dataset  
2. Handle missing values  
3. Encode categorical variables  
4. Normalize features if required  
5. Train Logistic Regression model  
6. Evaluate using accuracy & confusion matrix

---

## ▶️ How to Run Locally

1. **Clone the repository**
   - git clone https://github.com/Amankar003/Loan-Prediction.git
   cd Loan-Prediction

2. **Install dependencies**
   - pip install -r requirements.txt

3. **Launch Jupyter Notebook**
   - jupyter notebook Loan_Prediction.ipynb

---

 ## ✅ Future Improvements

* Add GUI using Streamlit or Flask
* Use advanced ML models (Random Forest, XGBoost)
* Deploy as a web service
* Perform hyperparameter tuning & cross-validation

---

## 🙌 Credits

* Dataset from open-source loan prediction data
* Modeling with Scikit-learn
* Visualizations with Matplotlib and Seaborn

---

## 🧑‍💻 Author

**Aman Kumar**

* GitHub: [@Amankar003](https://github.com/Amankar003)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
