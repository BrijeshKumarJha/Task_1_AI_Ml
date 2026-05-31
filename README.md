# 🚢 Titanic Survival Prediction: My First Machine Learning Model

## 📌 Project Overview
This project is a complete end-to-end Machine Learning pipeline built from scratch using a standard Python script. The goal is to predict the survival of passengers on the Titanic using a Logistic Regression model. 

Through this project, I practically implemented crucial Data Science concepts, starting from raw data ingestion to data cleaning, feature scaling, model training, and performance evaluation.

## 🛠️ Technologies & Libraries Used
* **Python 3.x**
* **Pandas:** Data manipulation and analysis
* **NumPy:** Numerical operations
* **Scikit-Learn:** Machine Learning algorithms and preprocessing tools
* **Matplotlib & Seaborn:** Data visualization and outlier detection

## 🚀 Key Learnings & Pipeline Steps

### 1. Data Loading & Exploration
Imported the raw Titanic dataset directly from a web source and analyzed its structure.

### 2. Handling Missing Values
* Dropped the `Cabin` column (too many nulls).
* Imputed missing `Age` values using the **Median**.
* Imputed missing `Embarked` values using the **Mode**.

![Missing Values Cleaning](Screenshots/Missing%20values%20cleaning.png)

### 3. Encoding Categorical Variables
* Applied **Label Encoding** to the `Sex` column (Male: 1, Female: 0).
* Applied **One-Hot Encoding** (`pd.get_dummies`) to the `Embarked` column.

### 4. Feature Scaling
Used **StandardScaler** to normalize numerical features (`Age` and `Fare`) so they have equal weight in the model.

![Age and Fare Standardization](Screenshots/Age%20and%20Fare%20Standardization.png)

### 5. Outlier Detection & Removal
Visualized `Fare` using Seaborn boxplots and removed extreme outliers using the **Interquartile Range (IQR)** method.

![Fare Outliers](Screenshots/Fare%20Outliers.png)

### 6. Train-Test Split
Divided the cleaned dataset into Training (80%) and Testing (20%) sets to ensure fair model evaluation.

### 7. Model Training
Trained a **Logistic Regression** algorithm on the processed data.

## 📊 Results
Evaluated the model using Accuracy Score and Confusion Matrix. The confusion matrix demonstrated a solid balance between True Positives and True Negatives, proving the model learned meaningful patterns rather than guessing blindly.

**Final Model Accuracy:** `75.48%`

![Model's Accuracy](Screenshots/Model's%20Accuracy.png)


