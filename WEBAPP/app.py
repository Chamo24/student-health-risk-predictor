import sys
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from model.preprocess import preprocess_data

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_health_model.pkl')
FEATURE_PATH = os.path.join(BASE_DIR, 'model', 'feature_columns.pkl')

try:
    model, label_encoder = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading error: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = {
            'id': [999],
            'step_count': [int(request.form['step_count'])],
            'exercise_duration': [float(request.form['exercise_duration'])],
            'sleep_duration': [float(request.form['sleep_duration'])],
            'bmi': [float(request.form['bmi'])],
            'heart_rate': [int(request.form['heart_rate'])],
            'calorie_expenditure': [float(request.form['calorie_expenditure'])],
            'gender': [request.form['gender']],
            'activity_level': [request.form['activity_level']],
        }

        df = pd.DataFrame(data)
        df = preprocess_data(df)

        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction_encoded = model.predict(df_encoded)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        try:
            proba = model.predict_proba(df_encoded)[0]
            confidence = round(max(proba) * 100, 2)
        except:
            confidence = 85.0

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)