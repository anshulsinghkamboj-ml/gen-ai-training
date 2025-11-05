from log import get_logger
import joblib
import pandas as pd
from save_load import save_file,load_latest_file
logger = get_logger(__name__)

latest_model = load_latest_file(base_dir="model", base_name="logistic_regression")
defaults = load_latest_file(base_dir='artifacts', base_name="default_feature_values")

user_input = {
    "mean radius": 15.2,
    "mean smoothness": 0.09
}

# Fill missing values with defaults
all_features = {**defaults, **user_input}
sample_df = pd.DataFrame([all_features])

# Predict
pred = latest_model.predict(sample_df)[0]
prob = latest_model.predict_proba(sample_df)[0]
print(f"Prediction: {pred}, Probabilities: {prob}")