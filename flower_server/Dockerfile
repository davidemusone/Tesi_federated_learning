# Utilizziamo Ubuntu 18.04 per avere compatibilità nativa con Python 3.6 e librerie legacy
FROM ubuntu:18.04

# Evita interazioni bloccanti durante l'apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Installazione di Python 3.6, pip e dipendenze di sistema necessarie per gRPC/C++
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Imposta python3.6 come python predefinito
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Aggiornamento setuptools e wheel (mantenendo pip compatibile)
RUN pip install --upgrade "pip<21.0" setuptools==59.6.0 wheel

# 1. Installazione precisa dei vincoli di base per evitare conflitti gRPC/Protobuf
RUN pip install \
    numpy==1.13.3 \
    protobuf==3.19.6 \
    grpcio==1.43.0 \
    grpcio-tools==1.43.0

# 2. Installazione di PyTorch e Torchvision
# Nota: La versione CPU basta per orchestrare e aggregare i pesi su Flower
RUN pip install torch==1.10.0 torchvision==0.11.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 3. Installazione di Flower 0.18.0 e dipendenze ausiliarie
RUN pip install flwr==0.18.0

# Imposta la directory di lavoro
WORKDIR /app

COPY server_app.py /app/server_app.py
COPY task.py /app/task.py

# Porta predefinita utilizzata da Flower per gRPC
EXPOSE 8080

CMD ["python", "server_app.py"]