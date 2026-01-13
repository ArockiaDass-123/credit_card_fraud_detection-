
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    data = pd.DataFrame([features])
    prediction = model.predict(data)[0]
    result = "⚠️ Fraud Detected!" if prediction == -1 else "✅ Transaction is Safe."
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
