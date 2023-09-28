from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Define hyperparameter grid for a hypothetical model
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200]
}

X_train = pd.read_csv('data/processed/X_train_engineered.csv')
X_test = pd.read_csv('data/processed/X_test_engineered.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Convert y_train and y_test to 1D arrays
y_train = y_train['Churn'].values
y_test = y_test['Churn'].values

# Initialize the model (Random Forest as an example)
model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :" , accuracy)

joblib.dump(model, 'docs/optimized_churn_prediction.pkl')