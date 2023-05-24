FROM python:3.9-slim-buster

EXPOSE 8001

WORKDIR /code/
COPY ./deploy_onnx/ /code/
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["gunicorn", "deploy:app", "-b", "0.0.0.0:8001", "-k", "uvicorn.workers.UvicornWorker"]
