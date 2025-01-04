FROM python:3.10.15-slim-bookworm

ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y tzdata
RUN ln -snf /user/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install -U poetry

WORKDIR /workdir
COPY poetry.lock pyproject.toml /workdir/

RUN chmod +x /usr/local/bin/poetry
RUN ls -l /usr/local/bin/poetry
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction 

ENV PYTHONPATH=/usr/local/bin/python3.10
 
RUN sed -i '1s|^.*$|#!/usr/local/bin/python3.10|' /usr/local/bin/poetry
 
COPY ./src /workdir/src
WORKDIR /workdir/src

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]