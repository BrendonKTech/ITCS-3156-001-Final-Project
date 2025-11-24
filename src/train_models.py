from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300, random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_gb(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05
    )
    model.fit(X_train, y_train)
    return model
