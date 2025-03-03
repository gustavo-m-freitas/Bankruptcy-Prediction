# 📊 Bankruptcy Prediction  
### Predicting Corporate Financial Distress with Machine Learning  

## Overview  
This project is a **Data Science portfolio** piece focused on **predicting corporate bankruptcy** using **financial data and machine learning techniques**. The goal is to develop a scalable and interpretable model to assess financial risk and help investors, analysts, and decision-makers prevent financial losses.  

This **Data Science portfolio project** is inspired by the methodology proposed in the research paper **"Financial Ratios and Corporate Governance Indicators in Bankruptcy Prediction: A Comprehensive Study"**, published in the **European Journal of Operational Research**. The study, conducted by **Deron Liang, Chia-Chi Lu, Chih-Fong Tsai, and Guan-An Shih** from **National Central University, Taiwan**, explores the use of financial ratios (FRs) and corporate governance indicators (CGIs) for predicting bankruptcy.  The **methodology employed in this portfolio** aligns with the structured approach of **Liang et al. (2016)** in predicting bankruptcy using **FRs**. The **feature selection, preprocessing techniques, and modeling strategies** are adapted to incorporate **best practices** from the study while ensuring **applicability to modern machine learning methods**.  

Additionally, this project compares **the original research-based model** with **other state-of-the-art machine learning models**, including **Random Forest, XGBoost, Support Vector Machines (SVM), and Multi-Layer Perceptron (MLP)**, to evaluate different approaches in bankruptcy prediction.  

---

## 📂 Project Structure  
This project is divided into **two main parts**:  

### ** Part I: Data Exploration & Preprocessing**  
1️ **Project Overview** → Explains the context and objectives of the study.  
2️ **Dataset Download & Loading** → How the data was obtained and loaded.  
3️ **Dataset Characteristics** → Structure, variables, and initial insights.  
4️ **Exploratory Data Analysis (EDA)** → Distribution, correlation, and visualizations.  

### ** Part II: Modeling & Evaluation**  
5️ **Feature Engineering & Data Preprocessing** → Normalization, balancing techniques.  
6️ **Modeling & Evaluation** → Machine Learning models and performance metrics.  
7️ **Conclusion** → Summary of insights and future directions.

---

## ⚖ Model Comparison: Research Paper vs. New Approaches  
This project evaluates two different **bankruptcy prediction strategies**:  

**Paper Model (Liang et al., 2016)**  
**Stratified Sampling + Feature Selection**  
**85% Recall (Detects More Bankruptcies)**  
**Lower False Negatives (~14-15%)**  

**New Models (SMOTE + Cross-Validation)**  
**Higher Accuracy (~95%) but Fails at Detecting Bankruptcies**  
**High False Negatives (~48-50%) → Nearly Half of Bankruptcies Are Missed**  
**Low Precision (~0.39 for XGBoost, 0.36 for RF)**  

🔎 **Key Finding:** The **paper’s methodology performs better in real-world financial risk assessment**, as it prioritizes **recall over accuracy**.  

---

## 🤖 Machine Learning Models Used
The following **classification algorithms** were implemented and evaluated:

- **Random Forest**
- **Multi-Layer Perceptron (MLP)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree (CART)**
- **Naive Bayes**

---

## 📉 Evaluation Metrics
The models were assessed using the following key **performance metrics**:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of predicted bankrupt companies that were actually bankrupt.
- **Recall (Sensitivity)**: Proportion of actual bankrupt companies correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall, balancing both.
- **AUC (Area Under the Curve)**: Measures the model’s ability to distinguish between classes.
- **Type I Error (False Negative Rate - FN Rate)**: Percentage of bankrupt companies incorrectly classified as non-bankrupt.
- **Type II Error (False Positive Rate - FP Rate)**: Percentage of non-bankrupt companies incorrectly classified as bankrupt.

---

## 🛠 Technologies & Tools  
- **Programming:** Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM)  
- **Visualization:** Matplotlib, Seaborn  
- **Feature Engineering:** Financial Ratios, Normalization (MinMaxScaler), Handling Imbalanced Data (SMOTE, Stratified Sampling)  
- **Modeling:** Logistic Regression, Random Forest, XGBoost, Neural Networks  
- **Evaluation Metrics:** AUC-ROC, Confusion Matrix, False Negative Rate  

---

## 📊 Dataset Information  
- **Source:** UCI Machine Learning Repository  
- **Time Period:** 1999 - 2009  
- **Attributes:** 96 financial and operational indicators  
- **Target Variable:** `Y` (1 = Bankrupt, 0 = Not Bankrupt)  

---

## Conclusion  
In **bankruptcy prediction**, minimizing **False Negatives (FN)** is more critical than achieving high accuracy. This study demonstrates how a **data-driven approach** can improve financial risk assessment, balancing **predictive power and interpretability**.  

**The research-based models (Liang et al., 2016) proved to be more effective in identifying bankruptcies, making them a better choice for real-world financial risk analysis.**  

---

## ⚠️ Important Note  
This project was developed as part of a **Data Science portfolio** to demonstrate the application of **machine learning techniques** in bankruptcy prediction. The results are **not intended to serve as conclusive financial advice** or a **scientific study**. The models were tested within the dataset’s constraints and do not account for external economic conditions, financial market fluctuations, or regulatory changes.  



