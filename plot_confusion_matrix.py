import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture_web_app\dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture_web_app\gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X_test)

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.title("Gesture Recognition Confusion Matrix")
plt.show()