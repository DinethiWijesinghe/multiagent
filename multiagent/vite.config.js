import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: ['rico-fails-their-from.trycloudflare.com', '.trycloudflare.com'],
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/universities': 'http://127.0.0.1:8000',
      '/auth': 'http://127.0.0.1:8000',
      '/user': 'http://127.0.0.1:8000',
      '/documents': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/ocr': 'http://127.0.0.1:8000',
      '/applications': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
    },
  },
  preview: {
    host: '0.0.0.0',
    allowedHosts: ['rico-fails-their-from.trycloudflare.com', '.trycloudflare.com'],
  },
})