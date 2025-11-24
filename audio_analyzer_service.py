from flask import Flask, request, jsonify
import numpy as np
import librosa
from keras.models import load_model
import os
from flask_cors import CORS
# Import 'tempfile' to securely handle temporary files
import tempfile

app=Flask(__name__)
CORS(app)

MODEL_PATH="speech_emotion_model.h5"
# It is assumed that 'speech_emotion_model.h5' exists in the same directory.
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
    print("Model input shape:", model.input_shape)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_features(file_path):
    """
    Extracts MFCC features from an audio file using librosa.
    """
    # librosa.load can automatically handle various formats (wav, webm, ogg, etc.)
    y,sr=librosa.load(file_path)
    # Extract 40 MFCC features
    mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T , axis=0)
    # Reshape features for the model (1, 40, 1)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on startup.'}), 500

    file_path = None # Initialize file_path

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Determine the file extension (e.g., '.webm', '.wav')
        file_extension = os.path.splitext(file.filename)[1]
        # Create a secure, unique temporary file path with the correct extension
        # 'delete=False' is used to ensure we control the deletion in the finally block
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
        file_path = temp_file.name
        temp_file.close() # Close the handle before saving with file.save()

        # Save the uploaded file to the unique temporary path
        file.save(file_path)

        try:
            # librosa will use the correct extension from the temp path to load the file
            features = extract_features(file_path)
            prediction = model.predict(features)
            predicted_label = labels[np.argmax(prediction)]

            # Successful prediction, return the result
            return jsonify({'emotion': predicted_label})

        except Exception as fe:
            # Log the specific feature extraction failure
            print(f"Feature extraction failed for file {file_path}: {fe}")
            # The original error message here will now correctly indicate the librosa failure
            return jsonify({'error': 'Prediction failed: Feature extraction failed (librosa): A common cause is trying to load a file with the wrong extension or an invalid audio format. The file type received was: ' + file_extension}), 500

    except Exception as e:
        # Log general application errors
        print(f"General error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the temporary file safely
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up temporary file {file_path}: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
