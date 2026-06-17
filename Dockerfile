# Dockerfile for the Unsloth MCP server (used by the Glama MCP listing).
FROM node:20-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY package*.json ./
RUN npm ci

# Copy the rest of the source and compile TypeScript to build/index.js.
COPY . .
RUN npm run build

# Start the MCP server (stdio transport).
CMD ["node", "build/index.js"]
