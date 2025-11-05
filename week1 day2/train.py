from data_loader import load_data
from scaler import scale
from split_data import split
from log import get_logger
from save_load import save_file,load_latest_file
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
save_file(model, base_dir="model", base_name="logistic_regression")

latest_model = load_latest_file(base_dir="model", base_name="logistic_regression")