FROM node:20-slim

WORKDIR /app

# Install curl for debugging
RUN apt-get update && apt-get install -y curl

# Copy package files first
COPY training_scripts/package*.json ./

# Install all dependencies
RUN npm install --also=dev

# Copy the rest of the source code
COPY training_scripts/ ./

# Build the project
RUN npm run build

# The command to run the server
CMD ["node", "dist/server.js"]
