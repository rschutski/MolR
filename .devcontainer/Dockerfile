ARG BASE_IMAGE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Sagemaker settings for notebook
ARG NB_USER="sagemaker-user"
ARG NB_UID=1000
ARG NB_GID=100

# basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    openssh-client \
    git \
    lshw \
    build-essential \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Add a sagemaker user (necessary for Notebook spaces)
RUN useradd --create-home --shell /bin/bash --gid "${NB_GID}" --uid ${NB_UID} ${NB_USER}
RUN mkdir /opt/venv && chown -R ${NB_USER}:${NB_GID} /opt/venv

# Prepare virtual env so poetry will install everything there
ENV PATH="/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv ${VIRTUAL_ENV}

# Poetry
ARG POETRY_VERSION=1.8.2
RUN pip3 install --upgrade pip && pip3 install poetry==${POETRY_VERSION}
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN chmod -R a+rw /opt/venv
USER ${NB_USER}

# Copy requirements and the source folder to the /code
WORKDIR /home/${NB_USER}
COPY pyproject.toml .
ADD src src

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Run the service (necessary for Notebook spaces)
ENTRYPOINT ["jupyter-lab"]
CMD ["--ServerApp.ip=0.0.0.0", "--ServerApp.port=8888", "--ServerApp.allow_origin=*", "--ServerApp.token=''", "--ServerApp.base_url=/jupyterlab/default"]                
