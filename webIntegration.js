import { useState } from 'react';

export default function Home() {
  const [predictedText, setPredictedText] = useState('');

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/translate', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    setPredictedText(data.predicted_text);
  };

  return (
    <div>
      <input type="file" onChange={handle}></input>
    </div>)}