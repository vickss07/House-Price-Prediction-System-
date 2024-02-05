# House-Price-Prediction-System-

This project focuses on building a housing price prediction model using machine learning techniques. The dataset used for training and testing the model is provided in the "housing.csv" file. The model is implemented in a Jupyter Notebook named "Housing_Price_Prediction.ipynb."

## Dependencies

- Python (>=3.6)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib

## Getting Started

1. Clone the repository:`git clone https://github.com/your-username/housing-price-prediction.git`.
2. Install the required dependencies:
   - numpy==1.21.3
   - pandas==1.3.4
   - matplotlib==3.4.3
   - seaborn==0.11.2
   - scikit-learn==0.24.2
   - joblib==1.1.0
   - pickle

3. Run the Jupyter Notebook: Open the "Housing_Price_Prediction.ipynb" file and execute the code cells.

## Data Preprocessing
- The dataset is loaded from the "housing.csv" file and missing values are dropped.
- Features are split into input (X) and target (y).
- The data is split into training and testing sets using the train_test_split method.

## Exploratory Data Analysis (EDA)
- Histograms and correlation matrices are generated to explore the dataset.
- Categorical data is one-hot encoded for better model performance.

## Model Training
- Linear Regression and Random Forest Regression models are trained on the preprocessed data.
- GridSearchCV is used to find the optimal hyperparameters for the Random Forest model.

## Model Evaluation
- The models are evaluated on the testing set using the R-squared score.
- Additional metrics such as MAE and RMSE are calculated for both Linear Regression and Random Forest models.

## Model Persistence
- The trained Random Forest model is saved using both Pickle and Joblib for future use.

## Predictions
- The saved model is loaded, and predictions are made on new data.

## Files
- Housing_Price_Prediction.ipynb: Jupyter Notebook containing the code.
- housing.csv: Dataset file (California).
- model_pickle: Pickle file containing the trained Random Forest model.
- House_Model_Joblib.joblib: Joblib file containing the trained Random Forest model.










