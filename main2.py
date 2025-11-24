import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')
import pygame

# ✅ Path to your dataset
dataset_path = "C:/Users/anupa/Desktop/speech emotion recognition/archive/TESS Toronto emotional speech set data"

# ✅ Collect all audio file paths and labels
paths = []
labels = []
for dirname, _, filenames in os.walk(dataset_path):
    filenames.sort()  # ensures consistent ordering
    for filename in filenames:
        if filename.endswith('.wav'):
            full_path = os.path.join(dirname, filename)
            label = filename.split('_')[-1].split('.')[0].lower()
            paths.append(full_path)
            labels.append(label)

print(f"✅ Dataset loaded: {len(paths)} files found\n")

# ✅ Create DataFrame
df = pd.DataFrame({
    'speech': paths,
    'label': labels
})

# ✅ Extract speaker type (OAF or YAF)
df['speaker'] = df['speech'].apply(lambda x: os.path.basename(x).split('_')[0])

print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())

print("\nSpeaker distribution:")
print(df['speaker'].value_counts())

# ✅ Plot label distribution
sns.countplot(x='label', data=df)
plt.title('LABEL DISTRIBUTION')
plt.savefig('label_distribution.png')
plt.close()

# ✅ Define helper functions
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.savefig(f'{emotion}_waveplot.png')
    plt.close()

def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig(f'{emotion}_spectrogram.png')
    plt.close()

def play_audio(path, duration=5):
    """Play .wav file using pygame for a limited duration."""
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"▶️ Playing: {os.path.basename(path)} for {duration} seconds")
    pygame.time.wait(duration * 1000)
    pygame.mixer.music.stop()

# ✅ Try playing samples from both speakers
try:
    emotion = 'fear'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print("\nLoaded OAF fear sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

try:
    emotion = 'angry'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

try:
    emotion = 'disgust'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

try:
    emotion = 'neutral'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

try:
    emotion = 'sad'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")


try:
    emotion = 'ps'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")


try:
    emotion = 'happy'
    subset = df[(df['label'] == emotion) & (df['speaker'] == 'OAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded OAF {emotion} sample:", path)
    waveplot(data, sr, emotion)
    spectrogram(data, sr, emotion)
    play_audio(path)

    subset = df[(df['label'] == emotion) & (df['speaker'] == 'YAF')].reset_index(drop=True)
    path = subset['speech'][0]
    data, sr = librosa.load(path)
    print(f"\nLoaded YAF {emotion} sample:", path)
    waveplot(data, sr, f"{emotion}_YAF")
    spectrogram(data, sr, f"{emotion}_YAF")
    play_audio(path)

except Exception as e:
    print(f"⚠️ An error occurred: {e}")

def extract_mfcc(filename):
    y,sr=librosa.load(filename,duration=3, offset=0.5)
    mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfcc
print("\nExtracting MFCC features...")

# For OAF
oaf_file = df[df['speaker'] == 'OAF'].reset_index(drop=True)['speech'][0]
oaf_features = extract_mfcc(oaf_file)
print("\nOAF MFCC features:\n", oaf_features)

# For YAF
yaf_file = df[df['speaker'] == 'YAF'].reset_index(drop=True)['speech'][0]
yaf_features = extract_mfcc(yaf_file)
print("\nYAF MFCC features:\n", yaf_features)

X_mfcc=df['speech'].apply(lambda x: extract_mfcc(x))
print(X_mfcc)

X=[x for x in X_mfcc]
X=np.array(X)
print(X.shape)

X=np.expand_dims(X,-1)
print(X.shape)

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse_output=False)
y=enc.fit_transform(df[['label']])

print(y.shape)
print("One-hot encoded labels matrix for all 5600 files:")
print(y)

# Example code for plotting accuracy (corrected)
# epochs = list(range(50))
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, label='train_accuracy')
# plt.plot(epochs, val_acc, label='val_accuracy')  # Fixed: use val_acc for val_accuracy
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()

from keras.models import Sequential
from keras.layers import Dense , LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')


]

)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history= model.fit(X,y ,validation_split= 0.2 , epochs=50, batch_size=64)
print(history)

epochs=list(range(50))
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
plt.plot(epochs,acc,label='train_accuracy')
plt.plot(epochs,val_acc,label='val_accuracy')  # Fixed: use val_acc for val_accuracy
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss=history.history['loss']
val_loss=history.history['val_loss']
plt.plot(epochs,loss,label='train_loss')
plt.plot(epochs,val_loss,label='val_loss')  # Fixed: use val_loss for val_loss
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# ✅ Save the trained model
model.save("speech_emotion_model.h5")

# ✅ Save label encoder for later use
import joblib
joblib.dump(enc, "label_encoder.pkl")

print("✅ Model and encoder saved!")
