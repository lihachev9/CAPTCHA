FROM python:3.8-slim-buster

EXPOSE 8000

WORKDIR /code/
COPY . /code/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["gunicorn", "deploy:app", "-b", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker"]
