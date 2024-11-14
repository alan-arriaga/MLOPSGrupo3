
# Generar la imagen a partir del Dockerfile

docker build -t mlops-api . 

# Ejecutar el contenedor con el API

docker run --rm -p 8000:8000 --name mlops-api mlops-api
