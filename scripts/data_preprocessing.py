import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('data/raw/customer_churn_large_dataset.csv')

# Initial data exploration (optional)
print(df.head()) 
print(df.info())  

# Handle missing data (if any)

df['Age'].fillna(df['Age'].mean(), inplace=True)


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Location'] = label_encoder.fit_transform(df['Location'])

# Split the data into features (X) and target (y)
X = df.drop(columns=['CustomerID', 'Name', 'Churn'])  
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
