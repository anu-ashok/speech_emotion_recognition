import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [audioURL, setAudioURL] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);

  // 🎙 Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Requesting 'audio/webm' which is a widely supported format 
      // for MediaRecorder that librosa can usually handle.
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current.ondataavailable = (e) => chunks.current.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        // Use the recorded MIME type from the recorder, or default to 'audio/webm'
        const blobType = mediaRecorderRef.current.mimeType || 'audio/webm';
        const blob = new Blob(chunks.current, { type: blobType });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        chunks.current = [];
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (e) {
      alert(`Could not start recording: ${e.message}. Check microphone permissions.`);
    }
  };

  // ⏹ Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handlePredict = async () => {
    if (!audioBlob) {
      alert("Please record or upload an audio file first.");
      return;
    }
    
    // Determine the correct file extension based on the blob's MIME type
    const mimeType = audioBlob.type;
    let fileExtension = 'webm'; // Default to webm
    if (mimeType.includes('wav')) {
      fileExtension = 'wav';
    } else if (mimeType.includes('ogg')) {
      fileExtension = 'ogg';
    }

    try {
      const formData = new FormData();
      // Pass the correct filename extension to the backend
      formData.append('file', audioBlob, `audio.${fileExtension}`); 
      
      // *** IMPORTANT: Make sure your Flask server is running on port 5000 as configured below ***
      const res = await fetch('http://127.0.0.1:5000/predict', { method: 'POST', body: formData });
      
      const data = await res.json();
      
      if (res.ok) {
        alert(`Predicted emotion: ${data.emotion}`);
      } else {
        // Display the specific error from the backend
        alert(`Prediction failed: ${data.error}`); 
      }
    } catch (error) {
      alert(`Request failed: ${error.message}. Is the backend running on http://127.0.0.1:5000?`);
    }
  };

  return(
    <div style={{textAlign: 'center', marginTop:'50px'}}>
      <h1>SPEECH EMOTION RECOGNITION</h1>
      {isRecording ? (
        <button onClick={stopRecording}>Stop Recording</button>
      ) : (
        <button onClick={startRecording}>Start Recording</button>
      )}
      {audioURL &&(
        <div style={{ marginTop:'50px'}}>
          <audio src={audioURL} controls></audio>
          <br/>
          <button onClick={handlePredict}>Predict Emotion</button>
        </div>
      )}

      <hr/>
      <h3>Upload a new  file</h3>
      <input type='file' accept='.wav, .mp3, .webm, .ogg' onChange={(e) => {
        const file = e.target.files[0];
        setAudioBlob(file);
        setAudioURL(URL.createObjectURL(file));
      }}/>
    </div>
  )
};

export default App;