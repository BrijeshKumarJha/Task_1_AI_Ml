import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "Titanic-Dataset.csv"
df = pd.read_csv(url)

print("Dataset info:")
print(df.info())

print("\nMissing values count")
print(df.isnull().sum())

df.drop(columns=['Cabin'], inplace=True)

df['Age'] = df["Age"].fillna(df["Age"].median())

df['Embarked'] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\n\n After Cleaning")
print(df.isnull().sum())

df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

df['Sex'] = df["Sex"].map({"male":1, "female":0})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\n", df.head())

scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print("\nAfter Standardization")
print(df[['Age', 'Fare']].head())

plt.figure(figsize=(10,5))
sns.boxplot(x=df['Fare'])

plt.title("Boxplot of Fare Outliers")
plt.show()

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df["Fare"] >= lower_bound) & (df["Fare"] <= upper_bound)]

print("\n\n")
print(f"Rows in original data {len(df)}")
print(f"Rows in data after removing outliers {len(df_clean)}")

y = df_clean['Survived']
X = df_clean.drop(columns=['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\n")
print(f"Total rows: {len(X)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

model = LogisticRegression()
model.fit(X_train, y_train)
print("\n\n Model is trained")

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print(f"Confusion matrix:\n{confusion_matrix(y_test, predictions)}")