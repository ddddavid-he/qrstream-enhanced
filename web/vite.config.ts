import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const wasmPkgDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  'wasm/pkg',
);

export default defineConfig({
  // Keep the existing deployment path for local/intranet builds. GitHub Pages
  // overrides this with /<repository>/ in the deployment workflow.
  base: process.env.QRSTREAM_BASE ?? '/web/qrstream-dev/v1.0.0/',
  plugins: [wasm()],
  resolve: {
    alias: {
      'qrstream-decode': wasmPkgDir,
    },
  },
  build: {
    target: 'es2022',
  },
  server: {
    // Camera requires a secure context; allow LAN testing over HTTPS proxies.
    host: true,
  },
});
