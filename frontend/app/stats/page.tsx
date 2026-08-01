'use client';

import { useEffect, useState } from 'react';
import { getStats } from '@/lib/api';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

export default function StatsPage() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    getStats().then(setStats);
  }, []);

  if (!stats) return <p className="text-gray-400 p-12">Loading...</p>;

  const pieData = [
    { name: 'Pneumonia', value: stats.pneumonia_count },
    { name: 'Normal',    value: stats.normal_count },
  ];

  return (
    <main className="max-w-3xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-semibold text-gray-900 mb-2">Statistics</h1>
      <p className="text-gray-500 mb-8">Aggregate data across all predictions.</p>

      {/* Stat cards */}
      <div className="grid grid-cols-2 gap-4 mb-10">
        <div className="border border-gray-200 rounded-xl p-6">
          <p className="text-sm text-gray-400 mb-1">Total scans</p>
          <p className="text-4xl font-bold text-gray-900">{stats.total}</p>
        </div>
        <div className="border border-gray-200 rounded-xl p-6">
          <p className="text-sm text-gray-400 mb-1">Avg confidence</p>
          <p className="text-4xl font-bold text-gray-900">{stats.avg_confidence}%</p>
        </div>
        <div className="border border-red-100 bg-red-50 rounded-xl p-6">
          <p className="text-sm text-red-400 mb-1">Pneumonia detected</p>
          <p className="text-4xl font-bold text-red-600">{stats.pneumonia_count}</p>
          <p className="text-sm text-red-400 mt-1">{stats.pneumonia_pct}% of total</p>
        </div>
        <div className="border border-green-100 bg-green-50 rounded-xl p-6">
          <p className="text-sm text-green-400 mb-1">Normal</p>
          <p className="text-4xl font-bold text-green-600">{stats.normal_count}</p>
          <p className="text-sm text-green-400 mt-1">{100 - stats.pneumonia_pct}% of total</p>
        </div>
      </div>

      {/* Pie chart */}
      <div className="border border-gray-200 rounded-xl p-6">
        <h2 className="text-lg font-medium text-gray-800 mb-4">Result distribution</h2>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}>
              <Cell fill="#dc2626" />
              <Cell fill="#16a34a" />
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </main>
  );
}