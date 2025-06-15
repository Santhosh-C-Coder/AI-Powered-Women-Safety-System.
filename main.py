import cv2
import librosa
from tensorflow.keras.models import load_model
from utils import preprocess_gesture, preprocess_audio
from alert_service import trigger_alert

# Load models
gesture_model = load_model('../model/gesture_model_cnn.h5')
voice_model = load_model('../model/voice_model_lstm.h5')

def detect_gesture(image_path):
    image = cv2.imread(image_path)
    processed = preprocess_gesture(image)
    pred = gesture_model.predict(processed)
    return pred.argmax()

def detect_voice(audio_path):
    processed = preprocess_audio(audio_path)
    pred = voice_model.predict(processed)
    return pred.argmax()

if __name__ == "__main__":
    gesture_result = detect_gesture('../data/sample_gesture.jpg')
    voice_result = detect_voice('../data/sample_audio.wav')

    print(f"Gesture Detection Result: {gesture_result}")
    print(f"Voice Detection Result: {voice_result}")

    if gesture_result == 1 or voice_result == 1:
        trigger_alert()
