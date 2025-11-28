FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pip setuptools wheel build scikit-build-core pybind11 numpy pytest

COPY . .

RUN python3 -m build --wheel

RUN pip install dist/*.whl

RUN python3 -m pytest test.py

CMD python3 -m build
