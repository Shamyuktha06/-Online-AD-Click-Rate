# 🧠 Online Ad Click Rate Prediction - IBM Challenge

## 🔍 Overview
This project predicts whether a user will click on an online advertisement based on features such as age, daily internet usage, device type, and more. This is a classification problem solved using supervised machine learning techniques.

## 🗂️ Dataset
Example features:
- `Daily Time Spent on Site`
- `Age`
- `Area Income`
- `Daily Internet Usage`
- `Ad Topic Line`
- `City`, `Country`
- `Male`
- `Clicked on Ad` (Target)

Dataset used: `ad_click_data.csv` (can be synthetic or from IBM dataset)

## 📊 ML Models Tried
- Logistic Regression ✅
- Random Forest ✅
- XGBoost (optional)

## 🧪 Evaluation Metrics
- Accuracy
- Precision, Recall
- Confusion Matrix
- ROC-AUC Score

## 🚀 How to Run
```bash
git clone https://github.com/your-username/ad_click_prediction.git
cd ad_click_prediction
pip install -r requirements.txt
```
Then open `ad_click_prediction.ipynb` and run all cells.

## ✅ Output
Model predicts if a user is likely to click on an ad (`0` or `1`).

## 📚 Libraries Used
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## 👤 Author
Made for IBM Hackathon / Challenge by [Your Name]
