FROM node:24.22.2-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./

# `npm ci` installs exactly what package-lock.json pins and refuses to run
# when it and package.json disagree, so the image can't drift from what was
# tested. (`npm install --legacy-peer-deps` used to hide a stale lockfile.)
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

COPY . .

RUN npm run build

FROM node:24.22.2-alpine AS runner

WORKDIR /app

COPY --from=builder --chown=node:node /app/.output ./.output

ENV NODE_ENV=production

USER node

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:3000/ || exit 1

CMD ["node", ".output/server/index.mjs"]
