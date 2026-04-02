import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(roi):
    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = np.expand_dims(roi.astype(np.float32), axis=[0, -1])
    interpreter.set_tensor(input_details[0]['index'], roi)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return EMOTIONS[np.argmax(output)]

st.title("😊 Real-Time Emotion Detection")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label = predict_emotion(roi)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)