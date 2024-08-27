# Usar una imagen base oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clonar el repositorio SpeechCraft
RUN git clone https://github.com/SocAIty/speechcraft.git /app/speechcraft

# Instalar las dependencias de Python
RUN pip install --upgrade pip
RUN pip install /app/speechcraft[full]
RUN pip install fastapi uvicorn

# Exponer el puerto que se utilizará para servir la aplicación
EXPOSE 8080

# Comando para iniciar la aplicación usando Uvicorn
CMD ["uvicorn", "speechcraft.server:start_server", "--host", "0.0.0.0", "--port", "8080"]
