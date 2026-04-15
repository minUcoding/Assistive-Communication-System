import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture sign project\dataset.csv")

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture sign project\gesture_model1.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))