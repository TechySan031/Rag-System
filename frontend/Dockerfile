# ===========================================================
#  Frontend Dockerfile — Multi-stage build
#  Stage 1: Build React app (Vite)
#  Stage 2: Serve with Nginx
# ===========================================================

# --- Build stage ---
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Build-time env: backend API URL (Nginx will proxy this)
ENV VITE_API_URL=/api

RUN npm run build

# --- Production stage ---
FROM nginx:alpine

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
