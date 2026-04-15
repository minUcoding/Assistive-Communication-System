from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter
import pyttsx3

app = Flask(__name__)

camera_on = False
current_gesture = ""
letters = []
buffer_size = 10
prediction_buffer = deque(maxlen=10)

dictionary = [
    "HELLO", "HELP", "HEY", "THANKS", "PLEASE", "YES", "NO", "GOOD", "BAD", "STOP", 
    "WATER", "FOOD", "GOOD MORNING", "HOW ARE YOU", "THANK YOU", "WELCOME", 
    "SORRY", "EXCUSE ME", "I AM FINE", "WHERE", "NAME", "FRIEND", "FAMILY"
]

translation = {
    "HELLO": "नमस्ते",
    "HELP": "मदद",
    "THANKS": "धन्यवाद",
    "THANK YOU": "शुक्रिया",
    "PLEASE": "कृपया",
    "YES": "हाँ",
    "NO": "नहीं",
    "GOOD": "अच्छा",
    "BAD": "बुरा",
    "STOP": "रुको",
    "WATER": "पानी",
    "FOOD": "खाना",
    "WELCOME": "स्वागत है",
    "GOOD MORNING": "शुभ प्रभात",
    "HOW ARE YOU": "आप कैसे हैं?",
    "SORRY": "माफ़ कीजिये",
    "I AM FINE": "मैं ठीक हूँ",
    "WHERE": "कहाँ",
    "NAME": "नाम",
    "FRIEND": "दोस्त"
}

last_added_time = 0
cooldown = 2

with open(r"C:\Users\user\OneDrive\Pictures\URMIN FILES\gesture project\gesture sign project\gesture_model1.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def generate_frames():
    global current_gesture, camera_on

    while True:
        if not camera_on:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                black_frame,
                "SYSTEM OFF",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )
            ret, buffer = cv2.imencode('.jpg', black_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
        
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        detected = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                    )
                hand_landmarks = results.multi_hand_landmarks[0]
                data = []

                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)
                data = np.array(data).reshape(1,-1)
                probs = model.predict_proba(data)[0]
                confidence = max(probs)
                prediction = model.classes_[probs.argmax()]

                if confidence < 0.80:
                    prediction = "None"
                if confidence > 0.75:
                    prediction_buffer.append(prediction)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                if len(prediction_buffer) == buffer_size:
                    detected = Counter(prediction_buffer).most_common(1)[0][0]
                    current_gesture = detected
                else:
                    detected = "None"
                cv2.putText(
                    frame,
                    f"Gesture: {prediction} | Confidence: {confidence*100:.1f}%",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            cv2.putText(
            frame,
            "Two Hand Detected",
            (20,150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )               
        if not results.multi_hand_landmarks:
            cv2.putText(
            frame,
            "No Hand Detected",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )      
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/suggestions")
def suggestions_api():
    word = "".join(letters).replace(" ", "").upper()
    if len(word)==0:
        return {"suggestions":[]}
    matches = [w for w in dictionary if w.startswith(word)]
    if word.upper() == "HEL":
        matches = ["HELLO", "HELP"]
    if word.upper() == "TH":
        matches = ["THANKS"]
    if word.upper() == "GO":
        matches = ["GOOD"]
    return {"suggestions":matches[:3]}
    
@app.route("/gesture")
def gesture():
    full_sentence = "".join(letters).strip()
    last_word = full_sentence.split()[-1].upper() if full_sentence else ""
    hindi = translation.get(last_word, "")
    return {
        "gesture": current_gesture,
        "word": "".join(letters),
        "hindi": hindi
    }

@app.route("/add_letter/<letter>")
def add_letter(letter):
    global letters
    if letter != "None":
        if letter == "SPACE":
            letters.append(" ")
        else:
            letters.append(letter)
    return {"word": "".join(letters)}

@app.route('/add/<word>')
def add_word(word):
    global letters
    letters.clear()       
    letters.append(word)  
    return {"sentence": word}

@app.route('/clear')
def clear():
    letters.clear()
    return {"word": ""}

@app.route("/speak/<text>")
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return {"status": "spoken"}

@app.route("/toggle_camera")
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return {"status": camera_on}

@app.route("/camera_off")
def camera_off():
    global camera_on
    camera_on = False
    return {"status": "off"}

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
