# Import libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Inspect dataset
print("Head of dataset:\n", df.head())
print("\nShape:", df.shape)
print("\nDescription:\n", df.describe())
print("\nMissing values:\n", df.isna().sum())

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[iris.feature_names], df['target'],
    test_size=0.2, random_state=42, stratify=df['target'])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
