from sklearn.preprocessing import StandardScaler
from log import get_logger
logger = get_logger(__name__)
def scale(X,y):
    logger.info('using std scaler....')
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    return X_scaled,y