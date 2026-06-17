import type { NextConfig } from "next";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Pin the workspace root to this frontend dir. A stray ~/package-lock.json makes
// Turbopack infer the home directory as the root, which breaks the dev HMR socket
// and chunk serving (blank page in `next dev`). Pinning it removes that ambiguity.
const projectRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: {
    root: projectRoot,
  },
  transpilePackages: [
    "@pipecat-ai/voice-ui-kit",
    "@pipecat-ai/client-js",
    "@pipecat-ai/client-react",
    "@pipecat-ai/small-webrtc-transport",
  ],
};

export default nextConfig;
