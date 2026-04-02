import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😊 Real-Time Emotion Detection")
st.write("Allow camera access and see your emotion detected live!")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48)) / 255.0
            roi = np.expand_dims(roi, axis=[0, -1])
            pred = model.predict(roi)
            label = EMOTIONS[np.argmax(pred)]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)