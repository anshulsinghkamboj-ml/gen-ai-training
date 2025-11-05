import joblib
import os
from datetime import datetime
from log import get_logger
logger = get_logger(__name__)
def save_file(obj, base_dir, base_name):
    """Save a Python object with timestamp-based versioning."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{base_name}_{timestamp}.pkl"
    file_path = os.path.join(base_dir, file_name)
    
    joblib.dump(obj, file_path)
    logger.info(f"File saved at {file_path}")
    return file_path

def load_latest_file(base_dir, base_name):
    """Load the most recent version of a saved object."""
    all_files = [
        f for f in os.listdir(base_dir)
        if f.startswith(base_name) and f.endswith(".pkl")
    ]
    
    if not all_files:
        logger.error(f"No saved versions found for {base_name} in {base_dir}")
        raise FileNotFoundError(f"No saved versions found for {base_name}")
    
    latest_file = sorted(all_files)[-1]  # last one is latest chronologically
    file_path = os.path.join(base_dir, latest_file)
    
    obj = joblib.load(file_path)
    logger.info(f"Loaded latest file: {file_path}")
    return obj