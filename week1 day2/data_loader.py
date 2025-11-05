from sklearn.datasets import load_breast_cancer
import pandas as pd
from save_load import save_file,load_latest_file
from log import get_logger
logger = get_logger(__name__)

def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X=pd.DataFrame(X, columns=data.feature_names)
    y=pd.array(y)
    logger.info(f"Loaded dataset with shape {X.shape}")
    return X,y

import joblib

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Save default feature means
defaults = X.mean().to_dict()
save_file(defaults,'artifacts', 'default_feature_values')


