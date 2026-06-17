import type { NextConfig } from "next";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Pin the workspace root to this frontend dir. A stray ~/package-lock.json makes
// Turbopack infer the home directory as the root, which breaks the dev HMR socket
// and chunk serving (blank page in `next dev`). Pinning it removes that ambiguity.
const projectRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "standalone",
  // Next 16 blocks /_next/* dev resources (HMR + JS chunks) from origins it
  // considers cross-origin. The app is reached at 127.0.0.1:3000, which Next
  // treats as different from its own "localhost" origin, so the bundle is
  // blocked and the page renders blank. Allow the local hosts explicitly.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
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
