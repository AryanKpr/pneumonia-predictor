const BASE_URL = 'http://localhost:8000';

export async function predictXray(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Prediction failed');
  return res.json();
}

export async function getPredictions(skip = 0, limit = 20) {
  const res = await fetch(`${BASE_URL}/predictions?skip=${skip}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch predictions');
  return res.json();
}

export async function getStats() {
  const res = await fetch(`${BASE_URL}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}