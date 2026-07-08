# Next.js frontend. Build stage compiles to a standalone server (enabled by
# `output: "standalone"` in next.config.ts); runtime stage ships only that.
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000
# Standalone output: server.js + minimal node_modules, plus static assets and
# public/ copied alongside it (the standalone bundle does not include them).
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# OpenShift arbitrary-UID (group 0) compatibility.
RUN chgrp -R 0 /app && chmod -R g=u /app

EXPOSE 3000
CMD ["node", "server.js"]
