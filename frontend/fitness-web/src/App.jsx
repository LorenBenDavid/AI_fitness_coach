import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert('Please select a video first!');

    const formData = new FormData();
    formData.append('video', file);

    setLoading(true);
    setResult(null);
    try {
      // Note: Make sure your Flask server is running on port 5000
      const res = await axios.post('http://127.0.0.1:5000/analyze', formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert('Analysis failed! Make sure the Backend server is running.');
    }
    setLoading(false);
  };

  return (
    <div
      style={{
        padding: '50px',
        textAlign: 'center',
        fontFamily: 'Arial',
        backgroundColor: '#f4f4f9',
        minHeight: '100vh',
      }}
    >
      <h1 style={{ color: '#333' }}>🏋️‍♀️ Powerlifting Form AI</h1>
      <p>Upload your workout video to check your form</p>

      <div style={{ margin: '20px 0' }}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            marginLeft: '10px',
            padding: '10px 20px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Analyzing... (Wait ~1 min)' : 'Analyze Video'}
        </button>
      </div>

      {result && (
        <div
          style={{
            marginTop: '30px',
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '10px',
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
            display: 'inline-block',
            minWidth: '300px',
          }}
        >
          <h2 style={{ borderBottom: '2px solid #eee', paddingBottom: '10px' }}>
            Result Summary
          </h2>
          <p>
            <strong>Exercise:</strong> {result.exercise}
          </p>
          <p>
            <strong>Technique Score:</strong> {result.score}%
          </p>
          <h3
            style={{ color: result.verdict === 'GOOD' ? '#28a745' : '#dc3545' }}
          >
            Verdict: {result.verdict}
          </h3>
        </div>
      )}
    </div>
  );
}

export default App;
