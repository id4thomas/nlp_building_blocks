FROM python:3.10.15-slim-bookworm

RUN pip install -U poetry

WORKDIR /workdir
COPY poetry.lock pyproject.toml /workdir/

RUN chmod +x /usr/local/bin/poetry
RUN ls -l /usr/local/bin/poetry
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction 

ENV PYTHONPATH=/usr/local/bin/python3.10
 
# Make poetry use installed python
RUN sed -i '1s|^.*$|#!/usr/local/bin/python3.10|' /usr/local/bin/poetry
 
# Copy Source
COPY ./src /workdir/src
WORKDIR /workdir/src