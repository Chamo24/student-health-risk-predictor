Student Health Risk Predictor
A web application that predicts a student's health condition (Healthy, At-Risk, Unhealthy) based on lifestyle data, using an optimized ensemble machine learning model. This project was developed as part of the CIS6005 Computational Intelligence module.

Kaggle Leaderboard Score: 0.94734

Quick Start
1. Install Dependencies
Make sure you have Python 3.8+ installed. Navigate to the WEBAPP directory in your terminal and install the required packages:

Bash
cd WEBAPP
pip install -r requirements.txt
2. Run the Application
Once the dependencies are installed, start the Flask web server:

Bash
python app.py
The application will be running on [http://127.0.0.1:5001](http://127.0.0.1:5001).

3. Access the Application
Open your web browser and go to the URL [http://127.0.0.1:5001](http://127.0.0.1:5001) to start making predictions.

Features
Machine Learning Powered: Utilizes a highly tuned ensemble model (CatBoost, XGBoost, LightGBM, Random Forest) for accurate predictions.

Multi-Class Prediction: Classifies health status into one of three distinct categories: Healthy, At-Risk, or Unhealthy.

Confidence Score: Provides a confidence percentage for each prediction to indicate the model's certainty.

Responsive UI: A clean, user-friendly single-page interface built with HTML, CSS, and Bootstrap.

Rigorous Data Pipeline: Implements a full preprocessing pipeline including feature engineering, outlier handling, and missing value imputation.

📁 Project Structure
Plaintext
.
├── Predicting_Student_Health_Risk.ipynb  # Google Colab Notebook for EDA & Model Training
├── submission.csv                        # Final Kaggle Leaderboard Submission
├── README.md                             # This documentation file
└── WEBAPP/                               # Main Web Application folder
    ├── app.py                            # Flask backend and API endpoint
    ├── requirements.txt                  # Python dependencies
    ├── model/                            # Saved ML model, encoder, and feature list
    ├── templates/                        # HTML file for the user interface
    └── static/                           # CSS and other static assets

Technologies Used
Development Environment: Google Colab (for Model Training) & Local Environment (for Web App)

Backend: Python, Flask

Machine Learning: Scikit-Learn, XGBoost, CatBoost, LightGBM, Optuna

Data Science: Pandas, NumPy, Matplotlib, Seaborn

Frontend: HTML5, CSS3, JavaScript, Bootstrap

Notes
The machine learning model was trained using Google Colab, and the exported model (best_health_model.pkl) is loaded locally in the Flask app.

The web application is for demonstration purposes only and should not be used for actual medical diagnosis.
