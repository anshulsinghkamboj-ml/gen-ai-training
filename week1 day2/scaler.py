from sklearn.preprocessing import StandardScaler
import pandas as pd
from log import get_logger

logger = get_logger(__name__)

def scale(X, y):
    logger.info("Using StandardScaler...")
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    return X_scaled, y, scaler
