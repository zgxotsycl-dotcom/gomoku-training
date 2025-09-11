FROM node:20-slim

WORKDIR /app

# Install curl for debugging purposes
RUN apt-get update && apt-get install -y curl

# Copy all project files
COPY . .

# Change to the correct directory before running npm commands
WORKDIR /app/training_scripts

# Install all dependencies and build the project
RUN npm install --also=dev
RUN npm run build

# The command to run the server
CMD ["node", "dist/server.js"]