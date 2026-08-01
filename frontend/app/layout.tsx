import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Pneumonia Detector',
  description: 'AI-powered chest X-ray analysis',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-900 antialiased">
        <nav className="border-b border-gray-200 px-6 py-4 flex gap-8">
          <a href="/" className="font-semibold text-blue-600">Pneumonia Detector</a>
          <a href="/history" className="text-gray-500 hover:text-gray-900 transition">History</a>
          <a href="/stats" className="text-gray-500 hover:text-gray-900 transition">Stats</a>
        </nav>
        {children}
      </body>
    </html>
  );
}