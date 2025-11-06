import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from data_loader import load_data
from split_data import split
from scaler import scale
from performance import performance
from save_load import save_file
from log import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


logger = get_logger(__name__)

def train_model():
    # 1️⃣ Load and preprocess data
    logger.info("Loading data...")
    X, y = load_data()

    logger.info("Scaling features...")
    X_scaled, y, scaler = scale(X, y)

    logger.info("Splitting into train/test...")
    X_train, X_test, y_train, y_test = split(X_scaled, y)

    # 2️⃣ Set up MLflow experiment
    mlflow.set_experiment("Breast_Cancer_LogisticRegression")

    with mlflow.start_run(run_name="logreg_experiment") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        # 3️⃣ Define and train the model
        model = LogisticRegression(max_iter=1000)
        logger.info("Training Logistic Regression model...")
        model.fit(X_train, y_train)
        logger.info("Training complete.")

        # 4️⃣ Evaluate performance
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        # 5️⃣ Log parameters and metrics to MLflow
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "solver": model.solver,
            "random_state": model.random_state,
        })
        mlflow.log_metrics(metrics)

        # 6️⃣ Save local artifacts (versioned)
        model_path = save_file(model, base_dir="model", base_name="logistic_regression")
        scaler_path = save_file(scaler, base_dir="artifacts", base_name="standard_scaler")

        # 7️⃣ Log artifacts and model to MLflow
        mlflow.log_artifact(model_path, artifact_path="local_model")
        mlflow.log_artifact(scaler_path, artifact_path="local_scaler")

        # Capture model signature for inference reproducibility
        logger.info("Logging model to MLflow with signature and input example...")
        input_example = pd.DataFrame(X_train[:1], columns=X_train.columns)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        logger.info("Artifacts and model logged successfully.")
        logger.info(f"Run {run_id} completed. View at MLflow UI.")

    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    train_model()
