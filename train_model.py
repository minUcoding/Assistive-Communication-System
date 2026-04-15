import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture sign project\dataset.csv", header=None, on_bad_lines='skip')

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Test accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model Accuracy:", acc)

# Save model
with open("gesture_model1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as gesture_model1.pkl")