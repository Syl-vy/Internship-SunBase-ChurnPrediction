import pandas as pd
from sklearn.preprocessing import StandardScaler


X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Perform feature engineering

X_train['Bill_Per_Usage'] = X_train['Monthly_Bill'] / X_train['Total_Usage_GB']
X_test['Bill_Per_Usage'] = X_test['Monthly_Bill'] / X_test['Total_Usage_GB']

# Apply feature scaling or normalization (StandardScaler in this case)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the engineered and scaled datasets
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train_engineered.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_engineered.csv', index=False)
