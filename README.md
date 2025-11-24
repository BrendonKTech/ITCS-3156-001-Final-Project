Machine Learning–Based Prediction of Urban Air Quality Using Italian Sensor Data

This project builds a full machine learning pipeline to predict carbon monoxide concentration (CO(GT)) using the well-known Air Quality Dataset collected in an Italian city. The dataset contains hourly averages of gas sensor signals, environmental conditions, and reference measurements from a certified analyzer.

The project implements:
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature scaling
- Three ML models (Linear Regression, Random Forest, Gradient Boosting)
- Model evaluation (RMSE, MAE, R²)
- Feature importance visualization
- Full end-to-end automation through main.py

The dataset used is the Air Quality Dataset, originally published by De Vito et al. (2008) and available on Kaggle:

https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set

It contains:
- 9,358 hourly measurements
- Metal-oxide gas sensor signals
- Temperature, humidity, and absolute humidity
- Certified reference analyzer values
- Missing data encoded as –200
- Target variable in this project: CO(GT).

Installation:
1. git clone https://github.com/BrendonKTech/ITCS-3156-001-Final-Project
2. cd air_quality_ml_project

3. python -m venv venv
4. source venv/bin/activate     # Mac/Linux
4. venv\Scripts\activate        # Windows

5. pip install -r requirements.txt
6. Make sure AirQualityUCI.csv is in the /data folder

7. To run: python main.py

Machine Learning Models Used:
1. Linear Regression
Baseline model for comparison.
Low computational cost but poor performance due to dataset non-linearity.

2. Random Forest Regressor
Handles non-linearity and noise better.
Shows lowest MAE, though R² is still negative.

3. Gradient Boosting Regressor
Highly sensitive to noise; overfits this dataset.
Shows poorest performance overall.

License:
This project is for educational use under the course ITCS-3156.