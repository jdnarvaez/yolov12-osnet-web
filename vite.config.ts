import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    viteStaticCopy({
      targets: [
        {
          // ORT constructs jsep/jspi WASM URLs dynamically at runtime so Vite
          // never sees them during bundling — they end up absent from dist.
          // Copy distribution to assets so ORT can find them
          src: 'node_modules/onnxruntime-web/dist/**',
          dest: 'assets',
        },
      ],
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  preview: {
    port: 4173,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
  },
  assetsInclude: ['**/*.onnx'],
  optimizeDeps: {
    exclude: [
      'onnxruntime-web',
      'onnxruntime-web/webgpu',
      'onnxruntime-web/webgl',
      'onnxruntime-web/wasm',
      'onnxruntime-web/cpu',
    ],
  },
  base: process.env.BASE_PATH || '/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    port: 4173,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
  },
});
