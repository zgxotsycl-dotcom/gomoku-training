FROM node:20-slim

WORKDIR /app

# Copy all files from the training_scripts directory
COPY . .

# Install dependencies and build
RUN npm install --also=dev
RUN npm run build

# Install curl for debugging
RUN apt-get update && apt-get install -y curl

CMD ["node", "dist/server.js"]
