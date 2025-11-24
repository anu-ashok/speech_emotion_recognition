# speech_emotion_recognition
#A machine learning project that recognizes human emotions (angry, sad, fear, happy, neutral, disgust, pleasant surprise) from speech using MFCC features, LSTM model, and a React + Flask web app.

#see only app.py(FLASK) , app.js(REACT FRONTEND) , main.py(LSTM MODEL TRAINED) files for codes 

#DEMO FEATURES


✔️ Record your own voice using the browser microphone
✔️ Upload .wav / .mp3 / .webm / .ogg audio files
✔️ Backend extracts MFCC features using Librosa
✔️ Deep learning model predicts 7 emotions
✔️ Live frontend built with React.js
✔️ Backend built with Flask (Python)

# MODEL ARCHITECTURE

#The LSTM model used:

#Input shape: (40 MFCCs, 1)

#LSTM Layer (256 units)

#Dense Layers: 128 → 64

#Activation: ReLU

#Dropout: 0.2

#Output Layer: Softmax (7 classes)



#link for dataset:- https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

