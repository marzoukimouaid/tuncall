import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to a local uvicorn on :8000; in Compose the API serves dist/.
const api = { target: "http://127.0.0.1:8000", changeOrigin: true };

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/calls": api, "/health": api } },
});
