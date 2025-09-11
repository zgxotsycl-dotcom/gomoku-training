# --- Stage 1: Build --- 
FROM node:20-slim as builder

WORKDIR /app

COPY . .

WORKDIR /app/training_scripts
RUN npm install --also=dev

# Run build and ignore errors for now to see the logs
RUN npm run build:cpu || true

# --- Stage 2: Production --- 
FROM node:20-slim

WORKDIR /app

COPY --from=builder /app/training_scripts/package.json ./package.json
COPY --from=builder /app/training_scripts/package-lock.json* ./package-lock.json*

RUN npm install --omit=dev

COPY --from=builder /app/training_scripts/dist ./dist

CMD ["node", "dist/server.js"]