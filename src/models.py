from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def get_linear_model(**kwargs):
    return LinearRegression(**kwargs)

def get_random_forest_model(**kwargs):
    return RandomForestRegressor(**kwargs)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test):
    return model.predict(X_test)
