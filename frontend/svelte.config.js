import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // SPA build — no SSR. Output is a static bundle that Vite serves in dev and
    // any static server (or FastAPI's StaticFiles) serves in production.
    adapter: adapter({
      fallback: 'index.html',
      strict: false
    })
  }
};

export default config;
