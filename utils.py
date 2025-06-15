import cv2
import numpy as np
import librosa

def preprocess_gesture(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return np.expand_dims(mfcc, axis=0)
