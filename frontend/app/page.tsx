'use client';

import { useState } from 'react';
import { predictXray } from '@/lib/api';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(f: File) {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictXray(file);
      setResult(data);
    } catch (e) {
      setError('Failed to analyze. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-semibold text-gray-900 mb-2">Pneumonia Detector</h1>
      <p className="text-gray-500 mb-8">Upload a chest X-ray to get an instant AI prediction.</p>

      {/* Upload zone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => document.getElementById('file-input')?.click()}
        className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition"
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="X-ray preview" className="mx-auto max-h-64 rounded-lg object-contain" />
        ) : (
          <div className="text-gray-400">
            <p className="text-lg">Drag & drop or click to upload</p>
            <p className="text-sm mt-1">JPG, PNG supported</p>
          </div>
        )}
      </div>

      {/* Analyze button */}
      {file && !result && (
        <button
          onClick={analyze}
          disabled={loading}
          className="mt-6 w-full bg-blue-600 text-white py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition"
        >
          {loading ? 'Analyzing...' : 'Analyze X-Ray'}
        </button>
      )}

      {error && <p className="mt-4 text-red-500 text-center">{error}</p>}

      {/* Result */}
      {result && (
        <div className="mt-8 space-y-6">
          <div className={`rounded-xl p-6 text-center ${result.result === 'PNEUMONIA' ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
            <p className="text-4xl font-bold mb-1" style={{ color: result.result === 'PNEUMONIA' ? '#dc2626' : '#16a34a' }}>
              {result.result}
            </p>
            <p className="text-gray-500 text-lg">Confidence: <span className="font-semibold text-gray-800">{result.confidence}%</span></p>
          </div>

          {/* Grad-CAM heatmap */}
          <div>
            <h2 className="text-lg font-medium text-gray-800 mb-2">What the model focused on</h2>
            <img
              src={`data:image/png;base64,${result.gradcam_b64}`}
              alt="Grad-CAM heatmap"
              className="w-full rounded-xl border border-gray-200"
            />
            <p className="text-sm text-gray-400 mt-2">Red/yellow regions indicate areas that influenced the prediction most.</p>
          </div>

          <button
            onClick={() => { setFile(null); setPreview(null); setResult(null); }}
            className="w-full border border-gray-300 text-gray-600 py-3 rounded-xl hover:bg-gray-50 transition"
          >
            Upload another
          </button>
        </div>
      )}
    </main>
  );
}