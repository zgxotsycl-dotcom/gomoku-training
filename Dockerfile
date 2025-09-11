# --- Stage 1: Build --- 
FROM node:20-slim as builder

WORKDIR /app

COPY . .

WORKDIR /app/training_scripts
RUN npm install --also=dev

# Directly run tsc to see the errors
RUN ./node_modules/typescript/bin/tsc --outDir dist --rootDir . --resolveJsonModule true --module commonjs --esModuleInterop true --target es2020 *.ts src/*.ts || true

# --- Stage 2: Production --- 
FROM node:20-slim

WORKDIR /app

COPY --from=builder /app/training_scripts/package.json ./package.json
COPY --from=builder /app/training_scripts/package-lock.json* ./package-lock.json*

RUN npm install --omit=dev

COPY --from=builder /app/training_scripts/dist ./dist

CMD ["node", "dist/server.js"]
