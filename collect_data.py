import cv2
import mediapipe as mp
import csv
import os

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Ask user for label
label = input("Enter gesture label (example: YES, STOP, A, B): ").upper()

file_name = r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture sign project\dataset.csv"
file_exists = os.path.isfile(file_name)

with open(file_name, mode='a', newline='') as f:
    writer = csv.writer(f)

    print("Show the gesture. Press Q to stop collecting.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                row.append(label)
                writer.writerow(row)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")