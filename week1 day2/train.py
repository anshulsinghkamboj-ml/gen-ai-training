from data_loader import load_data
from scaler import scale
from split_data import split
from log import get_logger
logger = get_logger(__name__)

X,y=load_data()
X_scaled,y=scale(X,y)
X_train, X_test, y_train, y_test=split(X_scaled,y)

from sklearn.linear_model import LogisticRegression
logger.info('training model')
model=LogisticRegression()
model.fit(X_train, y_train)

logger.info('doing predictions')
y_pred=model.predict(X_test)

from performance import performance
performance(y_pred,y_test)

#savingmodel
