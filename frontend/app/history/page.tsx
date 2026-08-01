'use client';

import { useEffect, useState } from 'react';
import { getPredictions } from '@/lib/api';

export default function HistoryPage() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getPredictions().then((data) => {
      setPredictions(data);
      setLoading(false);
    });
  }, []);

  return (
    <main className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-semibold text-gray-900 mb-2">Prediction History</h1>
      <p className="text-gray-500 mb-8">All past X-ray analyses, most recent first.</p>

      {loading ? (
        <p className="text-gray-400">Loading...</p>
      ) : predictions.length === 0 ? (
        <p className="text-gray-400">No predictions yet. Upload an X-ray on the home page.</p>
      ) : (
        <div className="space-y-4">
          {predictions.map((p) => (
            <div key={p.id} className="border border-gray-200 rounded-xl p-4 flex gap-6 items-start">
              {/* Heatmap thumbnail */}
              <img
                src={`data:image/png;base64,${p.gradcam_b64}`}
                alt="heatmap"
                className="w-24 h-24 rounded-lg object-cover border border-gray-100 shrink-0"
              />
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <span className={`text-lg font-bold ${p.result === 'PNEUMONIA' ? 'text-red-600' : 'text-green-600'}`}>
                    {p.result}
                  </span>
                  <span className="text-sm text-gray-400">Confidence: {p.confidence}%</span>
                </div>
                <p className="text-sm text-gray-400">
                  {new Date(p.created_at).toLocaleString()}
                </p>
              </div>
              {/* Confidence bar */}
              <div className="w-32 shrink-0">
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${p.result === 'PNEUMONIA' ? 'bg-red-400' : 'bg-green-400'}`}
                    style={{ width: `${p.confidence}%` }}
                  />
                </div>
                <p className="text-xs text-gray-400 mt-1 text-right">{p.confidence}%</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}