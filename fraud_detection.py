
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('dataset/creditcard.csv')

# Only use features for model
X = df.drop(columns=['Time', 'Class'])

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X)

# Predict anomalies
df['anomaly'] = model.predict(X)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = fraud

#pie chart
# Pie chart of fraud vs. non-fraud
labels = ['Normal', 'Fraud']
sizes = df['anomaly'].value_counts().sort_index()  # [0 = non-fraud, 1 = fraud]
colors = ['lightblue', 'lightcoral']
explode = (0, 0.1)  # only "explode" the fraud slice

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.2f%%', shadow=True, startangle=140)
plt.title('Fraud vs Normal Transaction Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

df['anomaly'] = model.predict(X)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = fraud


# Accuracy
print(classification_report(df['Class'], df['anomaly']))

# Save model
joblib.dump(model, 'model.pkl')
