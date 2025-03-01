# 📊 Bankruptcy Prediction  
### Predicting Corporate Financial Distress with Machine Learning  

## 🏆 Overview  
This project is a **Data Science portfolio** piece focused on **predicting corporate bankruptcy** using financial data and machine learning techniques. The goal is to develop a **scalable and interpretable model** to assess financial risk and help investors, analysts, and decision-makers prevent financial losses.  

We utilize financial statement indicators from the **Taiwan Economic Journal (1999-2009)**, applying **statistical analysis, feature engineering, and machine learning** to classify companies as **bankrupt or not bankrupt**.  

## 📂 Project Structure  
1️⃣ **Exploratory Data Analysis (EDA):** Identifying financial patterns and key risk indicators.  
2️⃣ **Feature Engineering:** Selecting relevant financial ratios and applying normalization techniques.  
3️⃣ **Machine Learning Models:** Implementing **Logistic Regression, Random Forest, XGBoost, and more**.  
4️⃣ **Model Evaluation:** Comparing **AUC-ROC, Precision-Recall, and False Negative Rate**.  
5️⃣ **Interpretability:** Using **SHAP values** to explain model decisions.  

---

## ⚖ Model Comparison: Research Paper vs. New Approaches  
This project evaluates two different **bankruptcy prediction strategies**:  

📖 **Paper Model (Liang et al., 2016)**  
✅ **Stratified Sampling + Feature Selection**  
✅ **85% Recall (Detects More Bankruptcies)**  
✅ **Lower False Negatives (~14-15%)**  

🚀 **New Models (SMOTE + Cross-Validation)**  
❌ **Higher Accuracy (~95%) but Fails at Detecting Bankruptcies**  
❌ **High False Negatives (~48-50%) → Nearly Half of Bankruptcies Are Missed**  
❌ **Low Precision (~0.39 for XGBoost, 0.36 for RF)**  

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
- **Future Scope:** API Deployment (Flask/FastAPI), Streamlit Dashboard  

---

## 📊 Dataset Information  
- **Source:** UCI Machine Learning Repository  
- **Time Period:** 1999 - 2009  
- **Attributes:** 96 financial and operational indicators  
- **Target Variable:** `Y` (1 = Bankrupt, 0 = Not Bankrupt)  

---

## 🚀 Conclusion  
In **bankruptcy prediction**, minimizing **False Negatives (FN)** is more critical than achieving high accuracy. This study demonstrates how a **data-driven approach** can improve financial risk assessment, balancing **predictive power and interpretability**.  

🔹 **The research-based models (Liang et al., 2016) proved to be more effective in identifying bankruptcies, making them a better choice for real-world financial risk analysis.**  


