# Stage 1: Build Flutter Web
FROM cirrusci/flutter:stable AS build

WORKDIR /app

# On copie d'abord les fichiers de configuration pour installer les dépendances
COPY pubspec.* ./
RUN flutter pub get

# On copie le reste du code source
COPY . .

# On compile l'application Web
RUN flutter build web

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copier les fichiers compilés de Flutter vers Nginx
COPY --from=build /app/build/web /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
