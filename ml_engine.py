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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def prepare_data(df, target_col):
    """
    Encodes categorical variables and splits features/target.
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y

def model_duel(df, target_col):
    """
    Tests multiple models and returns a performance comparison.
    """
    X, y = prepare_data(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    task_type = determine_task(df, target_col)
    results = []
    models = {}
    
    if task_type == 'classification':
        candidate_models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        for name, model in candidate_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_test, preds, average='weighted', zero_division=0)
            })
            models[name] = model
    else:
        candidate_models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Ridge Regression": Ridge()
        }
        for name, model in candidate_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results.append({
                "Model": name,
                "R-Squared": r2_score(y_test, preds),
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds))
            })
            models[name] = model

    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values(by=results_df.columns[1], ascending=False).iloc[0]['Model']
    
    return {
        "comparison": results_df,
        "best_model_name": best_model_name,
        "models": models,
        "X": X,
        "y": y
    }

def get_feature_importance(model, features):
    """
    Extracts feature importance from tree-based models or coefficients from linear ones.
    """
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_.flatten())
    else:
        return None
        
    return pd.DataFrame({'Feature': features, 'Importance': imp}).sort_values('Importance', ascending=False)

def predict_instance(model, instance_df):
    """
    Predicts outcome for a single data instance (What-If scenario).
    """
    return model.predict(instance_df)
