from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def evaluate(model, X_test, y_test, save_path):
    preds = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics, preds
