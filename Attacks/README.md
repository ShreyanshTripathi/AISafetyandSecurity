# TML25_A1_3

## 📌 Team 3 – Assignment 1 for the Trustworthy Machine Learning (TML) Course 2025

This repository contains our solution to **Assignment 1** of the **TML25** course. In this assignment, we developed and evaluated a machine learning model to perform adversarial attacks and analyze their impact.

---

## 📁 Repository Structure

| File / Folder                              | Description                                                                 |
|-------------------------------------------|-----------------------------------------------------------------------------|
| [`TML_Assignment_1.ipynb`](TML_Assignment_1.ipynb) | Main Jupyter notebook containing data loading, preprocessing, model training, and evaluation. |
| [`best_attack_model_catboost.pkl`](best_attack_model_catboost.pkl) | Saved CatBoost model used for adversarial attack generation or prediction. |
| [`test.csv`](test.csv)                     | Provided test data used for generating predictions or testing the model.   |
| [`TML_report.pdf`](TML_report.pdf)                     | Report file.   |

---

## 🧠 Summary Approach 

- We experimented with several models, including XGBOOST, CatBoost, AdaBoost, and MLP, and found that **CatBoost** performed the best in terms of robustness and predictive accuracy.

The plot below compares the ROC-AUC scores of different models we experimented with:

![ROC AUC Comparison](roc_auc_comparison.png)

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/TML25_A1_3.git
   cd TML25_A1_3

2. Load the model:
    ```bash
   import joblib
   model = joblib.load("best_attack_model_catboost.pkl")



# TML Assignment 1 Report  
**Team 3 - Hoang Nguyen Minh, Shreyansh Tripathi**  
*Date: [Today's Date]*

---

## Important Files  
Key components of this project:
1. [`simple_attack.py`](https://github.com/ShreyanshTripathi/TML25_A1_3/blob/main/simple_attack.py) - Simple attack implementation  
2. [`main_lira.py`](https://github.com/ShreyanshTripathi/TML25_A1_3/blob/main/main_lira.py) - LiRA method implementation  
3. [`main_rmia.py`](https://github.com/ShreyanshTripathi/TML25_A1_3/blob/main/main_rmia.py) - RMIA method implementation  
4. [`TML_Assignment_1.ipynb`](https://github.com/ShreyanshTripathi/TML25_A1_3/blob/main/TML_Assignment_1.ipynb) - Main analysis notebook

---

## Modeling Approach  
We implemented and compared three different attacks:

### 🎯 Simple Attack  
- Divided public dataset into train/test sets  
- Used features: entropy, loss, gradients, confidence  
- Compared four ML models:

| Model    | ROC-AUC  | Key Strength                |
|----------|----------|-----------------------------|
| CatBoost | 0.6645   | Handles categorical features|
| XGBoost  | 0.6613   | Strong generalization       |
| MLP      | 0.6474   | Non-linear pattern capture  |
| AdaBoost | 0.6440   | Simple ensemble             |

![ROC-AUC Comparison](roc_auc_comparison.png)

### 🔍 LiRA (Likelihood Ratio Attack)  
- 100-150 shadow models with matching architecture  
- Offline version with one-tailed hypothesis test  
- **Results**: TPR@FPR=0.05 ≈ 0.05, AUC ≈ 0.54  

### 📊 RMIA (Robust MIA)  
- 100-150 shadow models with reference dataset (n=3000)  
- Parameters: γ=2, a=0.2 (paper recommendations)  
- **Results**: TPR@FPR=0.05 ≈ 0.06, AUC ≈ 0.52  

### 🔄 Combined Attack  
- Integrated RMIA/LiRA confidence scores with simple features  
- Resulted in reduced performance  

---

## Conclusion  
**CatBoost** emerged as the most effective model with:
- Highest ROC-AUC (0.6645)  
- Automatic categorical feature handling  
- Robust regularization strategies  

---
