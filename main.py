import pandas as pd
from src.utils import load_data
from src.preprocess import clean_data, get_features_targets, split_and_scale
from src.train_models import train_linear, train_rf, train_gb
from src.evaluation import evaluate
from src.visualize import plot_feature_importance, plot_predictions

DATA_PATH = "data/AirQuality.csv"
TARGET = "CO(GT)"   # can change to Benzene, NOx, etc.

def main():
    # Load
    df = load_data(DATA_PATH)

    # Clean
    df = clean_data(df)

    # Features / target
    X, y = get_features_targets(df, TARGET)
    feature_names = X.columns

    # Split + scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # Models
    linear = train_linear(X_train, y_train)
    rf = train_rf(X_train, y_train)
    gb = train_gb(X_train, y_train)

    # Evaluate
    m1, pred1 = evaluate(linear, X_test, y_test, "results/metrics_linear.json")
    m2, pred2 = evaluate(rf, X_test, y_test, "results/metrics_rf.json")
    m3, pred3 = evaluate(gb, X_test, y_test, "results/metrics_gb.json")

    # Visualizations
    plot_feature_importance(rf, feature_names, "results/feature_importance_rf.png")
    plot_feature_importance(gb, feature_names, "results/feature_importance_gb.png")
    plot_predictions(y_test, pred2, "results/predictions_vs_true.png")

    print("\nLinear Regression:", m1)
    print("Random Forest:", m2)
    print("Gradient Boosting:", m3)

if __name__ == "__main__":
    main()
