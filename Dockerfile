# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app/
COPY . /app/