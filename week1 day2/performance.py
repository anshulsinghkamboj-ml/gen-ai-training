from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from log import get_logger
logger = get_logger(__name__)

def performance(y_pred,y_test):
    result=confusion_matrix(y_pred,y_test)
    logger.info(f"confusion matric : {result}")
    return result