import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle



df = pd.read_csv('data.csv')

print(df.head())
print(df.columns)
print(df.info())

print(df.describe())

df = df.dropna()

features = df.drop('Class',axis=1)
labels = df['Class']


print(features)
print(labels)

# # Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LogisticRegression())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
predictions = pipeline.predict(X_test)

# Evaluate the model
yhat = pipeline.predict(X_test)
print(yhat)
model_performance = classification_report(y_test,yhat)
print(f"Model Report: {model_performance}")

model_name = 'model.pkl'
with open(model_name,'wb')as f:
    pickle.dump(pipeline,f)

