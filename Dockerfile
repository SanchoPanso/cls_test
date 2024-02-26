FROM python:3.11.6
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install poetry==1.7.1

WORKDIR /app
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY cls /app/cls
COPY .env /app/
