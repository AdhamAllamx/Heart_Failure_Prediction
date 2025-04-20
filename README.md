# ❤️ Heart Failure Prediction using Machine Learning

## 📌 Overview

This project aims to build a machine learning pipeline that predicts the likelihood of heart disease based on patient data. Cardiovascular diseases (CVDs) are the leading cause of death globally, and early detection can significantly improve health outcomes. By leveraging data science, we seek to assist healthcare professionals in identifying high-risk individuals efficiently and accurately.

---

## 🧠 Problem Statement

Heart disease is a major global health concern. Many at-risk individuals remain undiagnosed until it's too late. Early intervention can save lives. Our goal is to use patient medical records to train and evaluate predictive models that can detect heart disease early.

---

## 🎯 Project Aim

- Build a classification model to detect heart disease.
- Use both unsupervised and supervised learning techniques.
- Evaluate model performance before and after dimensionality reduction.
- Tune hyperparameters to improve accuracy and generalization.
- Provide insights into feature importance for medical interpretability.

---

## 📊 Dataset Description

The dataset includes 11 key features about a patient’s health and the target variable `HeartDisease`.

| Feature         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Age            | Age of the patient (in years)                                               |
| Sex            | Gender of the patient (M/F)                                                 |
| ChestPainType  | Type of chest pain (TA, ATA, NAP, ASY)                                      |
| RestingBP      | Resting blood pressure (in mm Hg)                                           |
| Cholesterol    | Serum cholesterol level (in mg/dl)                                          |
| FastingBS      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                       |
| RestingECG     | ECG results (Normal, ST, LVH)                                               |
| MaxHR          | Maximum heart rate achieved                                                 |
| ExerciseAngina | Exercise-induced angina (Y/N)                                               |
| Oldpeak        | ST depression induced by exercise                                           |
| ST_Slope       | Slope of peak exercise ST segment (Up, Flat, Down)                          |
| HeartDisease   | Target class (1 = presence of heart disease, 0 = absence)                   |

---

## 🧱 Project Structure

1. Importing Required Libraries  
2. Loading Dataset into `heart_failure_df`  
3. Data Visualization & Exploration (Pre-cleaning)  
4. Data Cleaning & Processing:
   - Handling missing values, duplicates, outliers
   - Binning and smoothing
5. Feature Selection (Post-cleaning)
   - Correlation matrix, Chi-square test, and medical context
6. Normalization & Standardization using QQ Plot  
7. Feature Encoding:
   - One-hot encoding for categorical & binned features
8. Train-Test Split & Feature Scaling  
9. Dimensionality Reduction using PCA:
   - 2D visualization
   - 95% variance retention for modeling
10. Unsupervised Learning (Clustering):
    - KMeans clustering with elbow method, silhouette score, Dunn Index
    - Hierarchical clustering with dendrograms and Agglomerative Clustering
11. Supervised Learning (Classification):
    - Logistic Regression
    - Random Forest Classifier
    - Support Vector Machine (SVM)
    - Each evaluated with and without PCA
12. Model Comparison (With and Without PCA)
13. Decision Boundaries  
14. Hyperparameter Tuning:
    - K-Fold Cross Validation
    - GridSearchCV for model optimization
15. Feature Importance Analysis

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  
- Silhouette Score (for clustering)  
- Davies-Bouldin Index  
- Dunn Index  

---

## 🚀 Results

- PCA improved model interpretability while retaining performance.  
- Random Forest consistently performed best in terms of accuracy and interpretability.  
- Clustering results aligned with the labeled classification, validating unsupervised structure in the dataset.  
- Hyperparameter tuning significantly improved model generalization.  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Seaborn, Matplotlib  
- PCA, KMeans, AgglomerativeClustering  
- GridSearchCV, KFold  

---

## 📁 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/AdhamAllamx/heart-failure-prediction.git

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook Heart_Failure_Prediction.ipynb
```

---

## 🔮 Future Work

- Integrate more real-world medical datasets for better generalization  
- Deploy as a web app for doctors to input data and receive predictions  
- Incorporate deep learning models like CNNs on ECG images or signal data  

---

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- Medical literature on heart disease symptoms and risk factors  
- Scikit-learn documentation  

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

If you have any questions or feedback, feel free to contact [yourname@domain.com].

---

## 🧠 Acknowledgements

This project is a part of a machine learning coursework/project aiming to apply theory into impactful real-world applications.
