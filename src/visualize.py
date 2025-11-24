import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, save_path):
    if not hasattr(model, "feature_importances_"):
        return
    
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importance)), importance[idx])
    plt.xticks(range(len(importance)), np.array(feature_names)[idx], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True")
    plt.savefig(save_path)
    plt.close()
