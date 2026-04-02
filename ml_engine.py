import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

def determine_task(df, target_col):
    """
    Determines if the target is continuous (Regression) or categorical (Classification).
    """
    if pd.api.types.is_numeric_dtype(df[target_col]):
        unique_vals = df[target_col].nunique()
        if unique_vals < 20: 
            return 'classification'
        else:
            return 'regression'
    return 'classification'

def train_model(df, target_col):
    """
    Trains a model intelligently and returns predictions, metrics, and feature importances.
    """
    # Prepare data (simple Label Encoding for all categorical variables)
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    if X.empty:
        return {"error": "Not enough features to train a model."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    task_type = determine_task(df, target_col)

    model = None
    metric_name = ""
    score = 0

    try:
        if task_type == 'classification':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            # Default to RandomForest for robustness and easy feature importances without strict scaling
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            metric_name = "Accuracy"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            metric_name = "R-Squared (R2)"
    except Exception as e:
        return {"error": f"Model Training Failed: {str(e)}"}

    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    return {
        "task_type": task_type.capitalize(),
        "metric_name": metric_name,
        "score": score,
        "feature_importances": feature_imp_df
    }
