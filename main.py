import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Real-Time Emotion Detection")
run = st.checkbox("Start Webcam")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.expand_dims(roi, axis=[0, -1])
        pred = model.predict(roi)
        label = emotions[np.argmax(pred)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
```

---
