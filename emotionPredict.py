import os
import librosa
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

# Function to extract MFCC features
def extract_mfcc(sr, audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Load the trained model
model = load_model('New folder\\New folder\\emotion_model.h5')  # Make sure to use the correct path to your trained model

# Map numerical labels back to emotions
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Function to predict emotion
def predict_emotion(file_path):
    audio, sr = librosa.core.load(file_path, res_type='kaiser_best')
    features = extract_mfcc(sr, audio)
    test_data = np.array([features])

    prediction = model.predict(test_data)
    predicted_emotion = emotions[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Predicted Emotion: {predicted_emotion}")

# Take input from the user
your_audio_file_path = input("Enter the path to the audio file: ")
predict_emotion(your_audio_file_path)
