import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from models.logistic_model import train_lr
from models.decision_tree_model import train_dt
from models.xgboost_model import train_xgb

df = pd.read_csv("data/diabetic_data.csv")

drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.dropna(subset=['race', 'gender', 'age'])
df['race'] = df['race'].fillna('Unknown')
df['readmitted_flag'] = np.where(df['readmitted']=='<30', 1, 0)

y = df['readmitted_flag']
X = df.drop(columns=['readmitted','readmitted_flag'])

X_enc = pd.get_dummies(X, drop_first=True)
num_cols = X_enc.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y)

fpr_lr, tpr_lr, auc_lr = train_lr(X_train.values, X_test.values, y_train, y_test)
fpr_dt, tpr_dt, auc_dt = train_dt(X_train.values, X_test.values, y_train, y_test)
fpr_xgb, tpr_xgb, auc_xgb = train_xgb(X_train, X_test, y_train, y_test)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-криві для порівняння моделей')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
