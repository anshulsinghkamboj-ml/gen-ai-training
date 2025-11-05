from save_load import load_latest_file
import pandas as pd
from log import get_logger
logger = get_logger(__name__)

model = load_latest_file(base_dir="model", base_name="logistic_regression")
scaler = load_latest_file(base_dir="artifacts", base_name="standard_scaler")
defaults = load_latest_file(base_dir="artifacts", base_name="default_feature_values")

user_input = {"mean radius": 15.2, "mean smoothness": 0.09}
all_features = {**defaults, **user_input}
sample_df = pd.DataFrame([all_features])

# Scale with the same scaler
sample_scaled = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)

pred = model.predict(sample_scaled)[0]
prob = model.predict_proba(sample_scaled)[0]

logger.info(f"Predicted : {pred} with prob {prob} ")