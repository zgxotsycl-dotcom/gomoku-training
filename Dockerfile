# --- Stage 1: Build --- 
FROM node:20-slim as builder

# Set the working directory for the entire build stage
WORKDIR /app

# Copy all project files
COPY . .

# Run npm install and build inside the correct subfolder
RUN cd training_scripts && npm install --also=dev && npm run build:cpu

# --- Stage 2: Production --- 
FROM node:20-slim

WORKDIR /app

# Copy only production dependencies manifest
COPY --from=builder /app/training_scripts/package.json ./package.json
COPY --from=builder /app/training_scripts/package-lock.json* ./package-lock.json*

# Install only production dependencies
RUN npm install --omit=dev

# Copy the built application from the builder stage
COPY --from=builder /app/training_scripts/dist ./dist

CMD ["node", "dist/server.js"]
