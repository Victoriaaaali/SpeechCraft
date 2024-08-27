# Usa una imagen base de Python 3.10 slim
FROM python:3.10-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Instala las dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    && apt-get clean

# Crea un entorno virtual de Python
RUN python -m venv /app/venv

# Activa el entorno virtual y actualiza pip
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip

# Instala SpeechCraft desde PyPI con la opción full para incluir el API web
RUN pip install speechcraft[full]

# (Opcional) Clona el repositorio para obtener la versión más reciente de GitHub
# RUN git clone https://github.com/SocAIty/speechcraft.git /app/speechcraft
# RUN pip install /app/speechcraft

# Instala fairseq desde el archivo .whl proporcionado
RUN pip install https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl

# (Opcional) Instala PyTorch con soporte GPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Exponer el puerto para el servidor web
EXPOSE 8080

# Comando para ejecutar la aplicación usando Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
