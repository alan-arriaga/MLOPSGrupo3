# Utiliza la imagen base oficial de Python 3.10
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY ./requirements.txt /app/

# Instala las dependencias de Python desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "fastapi[standard]" uvicorn

# Copia el resto de la aplicación (si tienes archivos adicionales)
COPY . /app/

# Exponer el puerto que usarás para la aplicación (si es necesario)
EXPOSE 8000

# Comando por defecto para ejecutar cuando el contenedor inicie
CMD "fastapi" "run" "main.py"  # Cambia 'app.py' por el nombre de tu script principal
