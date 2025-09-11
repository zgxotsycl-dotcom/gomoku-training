FROM node:20-slim

WORKDIR /app

# Copy all project files
COPY . .

# Install all dependencies and build the project
RUN npm install --also=dev
RUN npm run build

# The command to run the server
CMD ["node", "dist/server.js"]
