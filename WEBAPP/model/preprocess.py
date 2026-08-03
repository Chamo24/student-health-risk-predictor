import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()
    
    # Feature engineering
    df['activity_score'] = df['step_count'] * df['exercise_duration'] / 1000
    df['sleep_efficiency'] = df['sleep_duration'] / df['sleep_duration'].max()
    df['bmi_heart_ratio'] = df['bmi'] * df['heart_rate']
    df['calorie_intensity'] = df['calorie_expenditure'] / (df['exercise_duration'] + 1)
    df['total_activity'] = df['step_count'] + df['exercise_duration'] * 100
    
    # Outlier clipping
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if col != 'id':
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
    
    # Missing values
    for col in num_cols:
        if col != 'id':
            df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'health_condition':
            df[col] = df[col].fillna('Unknown')
    
    return df