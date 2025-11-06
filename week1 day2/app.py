from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from save_load import load_latest_file
from log import get_logger
from pydantic import RootModel


logger = get_logger(__name__)

app=FastAPI(title="Breast Cancer Classifier API", version="1.0")

try:
    model = load_latest_file(base_dir="model", base_name="logistic_regression")
    scaler = load_latest_file(base_dir="artifacts", base_name="standard_scaler")
    defaults = load_latest_file(base_dir="artifacts", base_name="default_feature_values")
    logger.info("Artifacts loaded successfully at API startup.")
except Exception as e:
    logger.error(f"Error loading artifacts: {e}")
    raise RuntimeError("Failed to load artifacts.") from e


class FeatureInput(RootModel[dict]):
    pass

def predict_from_input(user_input: dict):
    # Merge user input with defaults
    all_features = {**defaults, **user_input}
    sample_df = pd.DataFrame([all_features])

    # Enforce same feature order as training
    try:
        sample_df = sample_df[model.feature_names_in_]
    except AttributeError:
        pass  # if model was trained without feature names

    # Scale using saved scaler
    sample_scaled = pd.DataFrame(
        scaler.transform(sample_df),
        columns=sample_df.columns
    )

    # Predict
    pred = int(model.predict(sample_scaled)[0])
    prob = model.predict_proba(sample_scaled)[0].tolist()

    class_labels = {0: "malignant", 1: "benign"}

    logger.info(
        f"Prediction complete | class={pred} ({class_labels[pred]}) | prob={prob}"
    )

    return {
        "predicted_class": pred,
        "label": class_labels[pred],
        "probabilities": {"malignant": prob[0], "benign": prob[1]},
    }

@app.post('/predict')
def predict(data: FeatureInput):
    try:
        user_input = data.root  # <-- note: `.root` not `.__root__`
        result = predict_from_input(user_input)
        return {
            "status": "success",
            "input_features": list(user_input.keys()),
            "result": result,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))