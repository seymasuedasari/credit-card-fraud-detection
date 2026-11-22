# Kredi Kartı Dolandırıcılık Tespiti

Makine öğrenmesi ile kredi kartı dolandırıcılığı tespiti projesi.

## Veri Seti
Kaggle — ULB Machine Learning Group  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
(creditcard.csv boyutu nedeniyle repoya eklenmemiştir, linkten indirin)

## Kurulum
pip install -r requirements.txt

## Çalıştırma
streamlit run app.py

## Sonuçlar
| Model | PR-AUC | F1 | Recall |
|-------|--------|-----|--------|
| Gradient Boosting | 0.8754 | 0.82 | 0.87 |
| LightGBM | 0.8744 | 0.90 | 0.86 |
| RF Tuned | 0.8728 | 0.93 | 0.86 |
| XGBoost | 0.8626 | 0.85 | 0.87 |
| Logistic Regression | 0.8205 | 0.54 | 0.89 |
