from data_loader import load_data
from split_data import split
from scaler import scale
from performance import performance
from save_load import save_file
from log import get_logger

from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)

def train_model():
    # 1️⃣ Load the dataset
    logger.info("Loading data...")
    X, y = load_data()

    # 2️⃣ Scale the data (and retrieve the fitted scaler)
    logger.info("Scaling data...")
    X_scaled, y, scaler = scale(X, y)

    # 3️⃣ Split data into train/test
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = split(X_scaled, y)

    # 4️⃣ Train the model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    # 5️⃣ Evaluate model performance
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    result = performance(y_pred, y_test)
    logger.info(f"Model performance (confusion matrix):\n{result}")

    # 6️⃣ Save the model and scaler with timestamps
    logger.info("Saving model and scaler...")
    save_file(model, base_dir="model", base_name="logistic_regression")
    save_file(scaler, base_dir="artifacts", base_name="standard_scaler")

    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_model()
